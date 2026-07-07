# INDUSTRIAL KNOWLEDGE INTELLIGENCE OS (IKIOS)

Built for ET AI Hackathon 2026 — Problem Statement 8: "AI for Industrial Knowledge
Intelligence: Unified Asset & Operations Brain."

## Project Vision

Not a chatbot. Not "chat with PDF." An Industrial Knowledge Intelligence Operating
System: ingest heterogeneous industrial documents, build structured organizational
knowledge, reason over it with specialized AI agents, and assist engineers with
maintenance, troubleshooting, compliance, and operational decision-making.

Every design decision prioritizes modularity, extensibility, explainability,
reliability, and maintainability — architected so it can evolve past the hackathon
into an open-source industrial AI platform.

## Current Phase: MVP (hackathon time-boxed)

The full target architecture is documented below and is the north star. **We are
not building all of it before we have a working demo.** Build order:

1. Ingestion → chunking → embeddings → Qdrant (working retrieval, no graph yet)
2. Entity/relation extraction → Neo4j (graph joins retrieval)
3. Planner + Retrieval + one specialist agent (Maintenance) via LangGraph
4. Reasoning stages (evidence → citation → answer) wired to the FastAPI chat endpoint
5. Frontend: upload, chat with citations, minimal graph view
6. Only then: Postgres (users/sessions), Redis (cache), MinIO (object storage),
   remaining specialist agents, report generation

Placeholders for deferred infra (Postgres/Redis/MinIO) are stubbed with local
SQLite / filesystem so the interfaces are correct from day one and swapping in the
real backend later is a config change, not a rewrite.

## Goals (full-scope target)

1. **Industrial Document Intelligence** — understand documents beyond OCR: equipment,
   components, failures, personnel, locations, standards, regulations, operating
   conditions, maintenance actions.
2. **Knowledge Graph Construction** — a living graph that evolves as documents arrive;
   every extracted entity becomes part of a structured graph, not an isolated embedding.
3. **Hybrid Retrieval** — BM25 + dense retrieval + graph traversal + metadata filtering
   + structured lookup, fused before reaching the LLM. Never vector-search-only.
4. **Agentic Reasoning** — specialized agents collaborate; the system reasons before
   answering; every answer is evidence-backed.
5. **Industrial Decision Support** — root cause analysis, predictive maintenance,
   compliance checking, incident investigation, knowledge discovery, report generation.

## Engineering Principles

SOLID, DRY, KISS, composition over inheritance, dependency injection, repository
pattern, service layer, strict typing, domain-driven design where practical. Never
optimize for less code — optimize for clarity and testability. No god classes, no
hardcoded business logic, no tight coupling. Every module independently replaceable.

## Architecture (six layers)

Ingestion → Knowledge Construction → Hybrid Retrieval → Agentic Intelligence →
Reasoning Engine → Analytics & Learning.

## Storage (target end-state; see Current Phase for MVP substitutions)

- **Postgres** — metadata, users, sessions, configuration
- **Neo4j** — knowledge graph: entities, equipment network, failure chains
- **Qdrant** — embeddings, semantic search
- **Redis** — caching, agent state, session memory
- **MinIO** — original files, images, scanned documents

## Document Ingestion Pipeline

Upload → File Classification → OCR → Layout Detection → Table Extraction → Figure
Extraction → Metadata Extraction → Chunk Generation → Entity Extraction → Knowledge
Graph Update → Embedding Generation → Storage.

Supported types (extensible): PDF, DOCX, XLSX, CSV, scanned images, emails,
engineering drawings, P&ID, inspection reports, maintenance logs, SOPs, OEM manuals,
shift logs, sensor exports. Preserve structure (headings, tables, captions, figures,
equipment IDs, cross-references) — never flatten to plain text.

## Knowledge Graph

Entities: Equipment, Valve, Pump, Motor, Bearing, Sensor, Pipe, Inspection,
Maintenance, Incident, Failure, Root Cause, Standard, Location, Personnel,
Department, Manufacturer, Permit, Procedure.

Relationships: CONNECTED_TO, INSTALLED_IN, REPORTED_IN, FAILED_DUE_TO, INSPECTED_BY,
MAINTAINED_BY, REFERENCES, PART_OF, CAUSES, RELATED_TO, REQUIRES. Every relationship
versioned.

## Hybrid Retrieval Pipeline

Intent Detection → Planner → Query Expansion → Metadata Filter → BM25 → Vector
Search → Knowledge Graph Traversal → Cross-Encoder Reranking → Context Builder → LLM.

## AI Agents

- **Planner** — decomposes requests, routes tasks. Never answers directly.
- **Retrieval** — collects evidence. No reasoning.
- **Maintenance** — maintenance records, OEM manuals, inspection reports, failure history.
- **Compliance** — Factory Act, ISO, OSHA, internal SOPs, quality standards.
- **Incident** — accident reports, near misses, historical failures, lessons learned.
- **Knowledge Graph** — graph queries; avoids embeddings unless necessary.
- **Report** — PDF reports, inspection/maintenance summaries, executive summaries.
- **Recommendation** — action plans; must justify every recommendation, never fabricate.

## Reasoning Engine

Planning → Evidence Collection → Evidence Verification → Knowledge Fusion →
Confidence Estimation → Answer Generation → Citation Verification → Final Response.

Every answer contains: Summary, Evidence, Confidence, Sources, Recommended Actions.

## Memory

Working (current conversation) / Semantic (vector DB) / Graph (Neo4j) / Long-term
(user history, preferences).

## Stack

- **Frontend**: React, TypeScript, Tailwind, React Query, React Flow (graph viz)
- **Backend**: FastAPI, Pydantic, SQLAlchemy, Alembic, LangGraph, Celery, Redis, Docker
- **LLM**: Qwen 3 or Llama 3.3 (configurable via `LLM_PROVIDER`)
- **Embeddings**: BGE-M3
- **Reranker**: BAAI bge-reranker-large
- **OCR**: PaddleOCR
- **Layout**: Docling
- **Entity Extraction**: GLiNER + LLM-assisted extraction
- **Knowledge Graph**: Neo4j
- **Vector DB**: Qdrant
- **Observability**: Langfuse, MLflow

## Project Structure

```
backend/app/
  api/            FastAPI routers
  core/           config, logging, security
  agents/         specialized agents
  orchestrator/   LangGraph orchestration
  retrieval/      BM25 + vector + graph fusion
  graph/          Neo4j access layer
  documents/      ingestion pipeline
  ocr/            OCR adapters
  embeddings/     embedding generation
  reranking/      cross-encoder reranking
  memory/         working/semantic/graph/long-term memory
  reasoning/      evidence fusion, confidence, citations
  workflows/      end-to-end orchestrated workflows
  models/         ORM / domain models
  repositories/   data access (repository pattern)
  schemas/        Pydantic request/response schemas
  services/       business logic / use cases
  tests/

frontend/src/
  components/
  pages/
  hooks/
  services/
  store/
```

## Coding Style

Strict typing everywhere. Never use `Any` unless unavoidable. Every public function
has a docstring. Every module has one clear responsibility. Favor composition.

## Milestones

M0 Scaffold/Docker/Config/Logging/Auth · M1 Ingestion/OCR/Parsing/Chunking · M2
Knowledge Graph (entities/relations/Neo4j) · M3 Hybrid Retrieval (vector/BM25/graph/
rerank) · M4 Agentic Intelligence (planner + specialist agents) · M5 Reasoning Engine
(evidence fusion/confidence/citations) · M6 Decision Support (RCA/maintenance/
compliance/incident) · M7 Analytics (monitoring/eval/dashboards).

## Rules

Never skip architecture. Never hardcode business logic. Never tightly couple
modules. Never sacrifice explainability for cleverness. Every AI output must be
traceable. Every recommendation must have supporting evidence. Every module remains
independently replaceable.
