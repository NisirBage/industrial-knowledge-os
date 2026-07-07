"""ML Laboratory API (M6): train real scikit-learn models on synthetic
labeled data, register them, swap them into the live PredictionEngine, and
expose registry/explainability/drift endpoints for the ML Lab frontend.

Only failure_prediction (classification) and maintenance_priority
(regression) have synthetic label generators wired up — see
app/ml/datasets/synthetic.py's docstring for why the other five MLTaskType
values aren't faked here.
"""

import time

from fastapi import APIRouter, HTTPException, Request
from sklearn.metrics import confusion_matrix, roc_curve

from app.ml.datasets.models import TASK_PROBLEM_TYPE, MLProblemType, MLTaskType
from app.ml.datasets.synthetic import generate_failure_prediction_dataset, generate_maintenance_priority_dataset
from app.ml.explainability.explainer import Explainer
from app.ml.inference.model import MLPredictionModel
from app.ml.monitoring.monitor import ModelMonitor
from app.ml.registry.repository import SqlModelRegistry, hash_dataframe
from app.ml.training.trainer import SklearnTrainer
from app.prediction.engine import PredictionEngine
from app.prediction.models import PredictionTaskType
from app.research.experiments.manager import ExperimentManager
from app.research.model_registry import prediction_version
from app.schemas.ml import (
    CalibrationPointOut,
    DriftRequest,
    DriftResultOut,
    FeatureDefinitionOut,
    FeatureImportanceOut,
    MLTrainingResultOut,
    MLTrainRequest,
    ModelRecordOut,
    RocPointOut,
)

router = APIRouter(prefix="/ml", tags=["ml"])

# task -> (synthetic generator, PredictionTaskType this model serves once trained)
_SYNTHETIC_GENERATORS = {
    MLTaskType.FAILURE_PREDICTION: (generate_failure_prediction_dataset, PredictionTaskType.FAILURE_RISK),
    MLTaskType.MAINTENANCE_PRIORITY: (generate_maintenance_priority_dataset, PredictionTaskType.MAINTENANCE_PRIORITY),
}


def _model_record_to_out(record) -> ModelRecordOut:
    return ModelRecordOut(
        model_id=record.model_id,
        task_type=record.task_type,
        algorithm=record.algorithm,
        training_data_hash=record.training_data_hash,
        hyperparameters=record.hyperparameters,
        metrics=record.metrics,
        feature_schema=record.feature_schema,
        timestamp=record.timestamp,
        author=record.author,
        experiment_id=record.experiment_id,
        serialization_path=record.serialization_path,
        serialization_format=record.serialization_format,
    )


@router.post("/train/{ml_task_type}", response_model=MLTrainingResultOut)
async def train_model(ml_task_type: str, payload: MLTrainRequest, request: Request) -> MLTrainingResultOut:
    try:
        task_type = MLTaskType(ml_task_type)
    except ValueError:
        valid = ", ".join(t.value for t in MLTaskType)
        raise HTTPException(status_code=422, detail=f"Unknown ml_task_type '{ml_task_type}'. Valid values: {valid}")

    if task_type not in _SYNTHETIC_GENERATORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{ml_task_type}' has no synthetic label generator wired up yet "
                f"(see app/ml/datasets/synthetic.py). Trainable now: {[t.value for t in _SYNTHETIC_GENERATORS]}."
            ),
        )

    start = time.perf_counter()
    generator_fn, serving_task_type = _SYNTHETIC_GENERATORS[task_type]
    problem_type = TASK_PROBLEM_TYPE[task_type]

    dataset = generator_fn(payload.sample_size, payload.seed)
    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    trainer: SklearnTrainer = request.app.state.ml_trainer
    trained = trainer.train(X, y, problem_type, task_type.value, random_state=payload.seed)

    explainer: Explainer = request.app.state.ml_explainer
    feature_importance = explainer.feature_importance(trained.estimator, trained.feature_names)

    calibration: list[CalibrationPointOut] = []
    confusion: list[list[int]] | None = None
    roc_points: list[RocPointOut] = []
    if problem_type == MLProblemType.CLASSIFICATION:
        calibration_points = explainer.calibration_curve(trained.estimator, trained.X_test, trained.y_test)
        calibration = [CalibrationPointOut(predicted_probability=c.predicted_probability, observed_frequency=c.observed_frequency) for c in calibration_points]

        y_pred = trained.estimator.predict(trained.X_test)
        confusion = confusion_matrix(trained.y_test, y_pred).tolist()

        if hasattr(trained.estimator, "predict_proba"):
            probabilities = trained.estimator.predict_proba(trained.X_test)[:, 1]
            fpr, tpr, _thresholds = roc_curve(trained.y_test, probabilities)
            roc_points = [RocPointOut(fpr=float(f), tpr=float(t)) for f, t in zip(fpr.tolist(), tpr.tolist(), strict=True)]

    training_data_hash = hash_dataframe(dataset)

    experiment_manager: ExperimentManager = request.app.state.experiment_manager
    experiment = await experiment_manager.start(
        name=f"ml_training:{task_type.value}",
        description=f"Train {trained.algorithm} for {task_type.value} on a synthetic dataset",
        configuration={"sample_size": str(payload.sample_size), "seed": str(payload.seed)},
        model_versions=[prediction_version(trained.algorithm, "v1")],
        retriever_versions=[],
        graph_versions=[],
        reranker=request.app.state.settings.reranker_model,
        ontology_version="1.0",
        dataset="synthetic",
        random_seed=payload.seed,
    )
    await experiment_manager.complete(experiment.experiment_id, trained.metrics)

    model_registry: SqlModelRegistry = request.app.state.model_registry
    record = await model_registry.register(
        estimator=trained.estimator,
        task_type=task_type.value,
        algorithm=trained.algorithm,
        training_data_hash=training_data_hash,
        hyperparameters=trained.hyperparameters,
        metrics=trained.metrics,
        feature_schema=trained.feature_names,
        author="api",
        experiment_id=experiment.experiment_id,
    )

    ml_model = MLPredictionModel(serving_task_type, trained.estimator, record, explainer)
    prediction_engine: PredictionEngine = request.app.state.prediction_engine
    prediction_engine.register_model(serving_task_type, ml_model)

    return MLTrainingResultOut(
        model=_model_record_to_out(record),
        feature_importance=[FeatureImportanceOut(feature_name=f.feature_name, importance=f.importance) for f in feature_importance],
        calibration_curve=calibration,
        confusion_matrix=confusion,
        roc_curve=roc_points,
        sample_size=payload.sample_size,
        duration_seconds=round(time.perf_counter() - start, 4),
    )


@router.get("/models", response_model=list[ModelRecordOut])
async def list_models(request: Request, task_type: str | None = None) -> list[ModelRecordOut]:
    model_registry: SqlModelRegistry = request.app.state.model_registry
    records = await model_registry.list(task_type)
    return [_model_record_to_out(r) for r in records]


@router.get("/models/{model_id}", response_model=ModelRecordOut)
async def get_model(model_id: str, request: Request) -> ModelRecordOut:
    model_registry: SqlModelRegistry = request.app.state.model_registry
    record = await model_registry.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No registered model with id {model_id!r}")
    return _model_record_to_out(record)


@router.get("/feature-definitions", response_model=list[FeatureDefinitionOut])
async def get_feature_definitions(request: Request) -> list[FeatureDefinitionOut]:
    feature_store = request.app.state.feature_store
    return [FeatureDefinitionOut(**vars(d)) for d in feature_store.feature_definitions()]


@router.post("/drift", response_model=DriftResultOut)
async def compute_drift(payload: DriftRequest, request: Request) -> DriftResultOut:
    monitor: ModelMonitor = request.app.state.model_monitor
    psi = monitor.prediction_drift(payload.baseline, payload.recent)
    return DriftResultOut(psi=psi, severity=monitor.drift_severity(psi))
