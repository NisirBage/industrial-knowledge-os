"""Vision Workspace API (M7): upload/list/analyze images and multimodal
query — the API surface the Vision Workspace frontend and the Multimodal
Retriever's HTTP consumers read from. Entirely independent of the chat
orchestrator's request path, following every prior milestone's
independent-subsystem convention: its own service/repository/router, no
changes to app/orchestrator or the M3 pipeline.
"""

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.vision.ingestion.models import VisionSourceType
from app.vision.ingestion.service import UnsupportedImageError
from app.schemas.vision import (
    DefectOut,
    DetectionOut,
    DiagramSymbolOut,
    GaugeReadingOut,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    OCRRegionOut,
    VisionAnalysisOut,
    VisionDocumentOut,
)
from app.retrieval.explain import explain_evidence
from app.schemas.chat import EvidenceOut

router = APIRouter(prefix="/vision", tags=["vision"])


def _bbox_list(bounding_box) -> list[int]:
    return [bounding_box.x_min, bounding_box.y_min, bounding_box.x_max, bounding_box.y_max]


def _document_to_out(document) -> VisionDocumentOut:
    return VisionDocumentOut(
        id=document.id,
        filename=document.filename,
        source_type=document.source_type.value,
        content_type=document.content_type,
        width=document.width,
        height=document.height,
        size_bytes=document.size_bytes,
        equipment_hint=document.equipment_hint,
    )


@router.post("/images", response_model=VisionDocumentOut)
async def upload_image(
    request: Request,
    file: UploadFile,
    source_type: str = Form(...),
    equipment_hint: str | None = Form(None),
) -> VisionDocumentOut:
    try:
        parsed_source_type = VisionSourceType(source_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unsupported source_type {source_type!r}. Valid: {[t.value for t in VisionSourceType]}")

    image_bytes = await file.read()
    ingestion_service = request.app.state.vision_ingestion_service
    try:
        document = await ingestion_service.ingest(
            image_bytes,
            filename=file.filename or "upload.png",
            content_type=file.content_type or "application/octet-stream",
            source_type=parsed_source_type,
            equipment_hint=equipment_hint or None,
        )
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _document_to_out(document)


@router.get("/images", response_model=list[VisionDocumentOut])
async def list_images(request: Request) -> list[VisionDocumentOut]:
    repository = request.app.state.vision_metadata_repository
    documents = await repository.list()
    return [_document_to_out(d) for d in documents]


@router.get("/images/{image_id}", response_model=VisionDocumentOut)
async def get_image(image_id: str, request: Request) -> VisionDocumentOut:
    repository = request.app.state.vision_metadata_repository
    document = await repository.get(image_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No vision image with id {image_id!r}")
    return _document_to_out(document)


@router.get("/images/{image_id}/file")
async def get_image_file(image_id: str, request: Request) -> FileResponse:
    """Serves the stored image bytes — the Vision Workspace's Interactive
    Viewer (M7.1) renders this directly in an <img> tag beneath its bounding-
    box overlays. Discovered missing during frontend integration: every other
    route returns metadata/analysis *about* an image, but nothing served the
    image itself."""
    repository = request.app.state.vision_metadata_repository
    document = await repository.get(image_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No vision image with id {image_id!r}")
    return FileResponse(document.storage_path, media_type=document.content_type)


@router.post("/images/{image_id}/analyze", response_model=VisionAnalysisOut)
async def analyze_image(image_id: str, request: Request) -> VisionAnalysisOut:
    """Runs OCR + diagram parsing + equipment detection + gauge reading +
    defect detection over the stored image, writes every result into the
    Vision Knowledge Graph (never returned-only), and echoes the results back
    for the Vision Workspace to render as overlays."""
    repository = request.app.state.vision_metadata_repository
    document = await repository.get(image_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No vision image with id {image_id!r}")

    ocr_provider = request.app.state.vision_ocr_provider
    diagram_parser = request.app.state.vision_diagram_parser
    object_detector = request.app.state.vision_object_detector
    gauge_reader = request.app.state.vision_gauge_reader
    defect_detector = request.app.state.vision_defect_detector
    graph_builder = request.app.state.vision_graph_builder

    ocr_regions = ocr_provider.extract_text(document.storage_path, image_id=document.id)
    diagram_symbols = diagram_parser.parse(document.storage_path, image_id=document.id)
    detections = object_detector.detect(document.storage_path, image_id=document.id, equipment_hint=document.equipment_hint)
    gauge_readings = gauge_reader.read_all(document.storage_path, image_id=document.id)
    defects = defect_detector.detect(document.storage_path, image_id=document.id)

    image_node_id = graph_builder.add_image(document)
    ocr_node_ids = graph_builder.add_ocr_regions(image_node_id, ocr_regions)
    diagram_node_ids = graph_builder.add_diagram_symbols(image_node_id, diagram_symbols)
    detection_node_ids = graph_builder.add_detections(image_node_id, detections)
    gauge_node_ids = graph_builder.add_gauge_readings(image_node_id, gauge_readings)
    defect_node_ids = graph_builder.add_defects(image_node_id, defects)

    return VisionAnalysisOut(
        image_id=document.id,
        ocr_regions=[
            OCRRegionOut(text=r.text, confidence=r.confidence, bounding_box=_bbox_list(r.bounding_box), graph_node_id=node_id)
            for r, node_id in zip(ocr_regions, ocr_node_ids, strict=True)
        ],
        diagram_symbols=[
            DiagramSymbolOut(symbol_type=s.symbol_type.value, confidence=s.confidence, bounding_box=_bbox_list(s.bounding_box), graph_node_id=node_id)
            for s, node_id in zip(diagram_symbols, diagram_node_ids, strict=True)
        ],
        detections=[
            DetectionOut(equipment_type=d.equipment_type, confidence=d.confidence, bounding_box=_bbox_list(d.bounding_box), graph_node_id=node_id)
            for d, node_id in zip(detections, detection_node_ids, strict=True)
        ],
        gauge_readings=[
            GaugeReadingOut(
                needle_angle_degrees=r.needle_angle_degrees,
                normalized_value=r.normalized_value,
                confidence=r.confidence,
                valid=r.valid,
                validation_note=r.validation_note,
                bounding_box=_bbox_list(r.bounding_box),
                graph_node_id=node_id,
            )
            for r, node_id in zip(gauge_readings, gauge_node_ids, strict=True)
        ],
        defects=[
            DefectOut(defect_type=d.defect_type.value, confidence=d.confidence, severity=d.severity, bounding_box=_bbox_list(d.bounding_box), graph_node_id=node_id)
            for d, node_id in zip(defects, defect_node_ids, strict=True)
        ],
    )


@router.post("/query", response_model=MultimodalQueryResponse)
async def multimodal_query(payload: MultimodalQueryRequest, request: Request) -> MultimodalQueryResponse:
    retriever = request.app.state.multimodal_retriever
    evidence = retriever.retrieve(payload.query, top_k=payload.top_k)
    return MultimodalQueryResponse(
        evidence=[
            EvidenceOut(
                chunk_id=e.chunk_id,
                document_id=e.document_id,
                text=e.text,
                page_number=e.page_number,
                entity_ids=e.entity_ids,
                relation_ids=e.relation_ids,
                graph_score=e.graph_score,
                vector_score=e.vector_score,
                bm25_score=e.bm25_score,
                reranker_score=e.reranker_score,
                source=e.source,
                confidence=e.confidence,
                agent=e.agent,
                ltr_score=e.ltr_score,
                final_rank=rank,
                explanation=explain_evidence(e, rank),
            )
            for rank, e in enumerate(evidence, start=1)
        ]
    )
