"""ExperimentManager: the API every benchmark/A-B-test/offline-evaluation run
goes through to record itself. Captures the git commit automatically (best
effort — silently omitted if this isn't a git checkout or git isn't on PATH,
never fails the experiment over a missing commit hash).
"""

import subprocess

from app.research.experiments.models import Experiment
from app.research.experiments.repository import ExperimentTrackerProtocol
from app.research.model_registry import ComponentVersion


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


class ExperimentManager:
    def __init__(self, tracker: ExperimentTrackerProtocol):
        self._tracker = tracker

    async def start(
        self,
        name: str,
        description: str,
        configuration: dict[str, str],
        model_versions: list[ComponentVersion],
        retriever_versions: list[ComponentVersion],
        graph_versions: list[ComponentVersion],
        reranker: str,
        ontology_version: str,
        dataset: str,
        random_seed: int | None = None,
    ) -> Experiment:
        return await self._tracker.create(
            name=name,
            description=description,
            configuration=configuration,
            model_versions=model_versions,
            retriever_versions=retriever_versions,
            graph_versions=graph_versions,
            reranker=reranker,
            ontology_version=ontology_version,
            dataset=dataset,
            random_seed=random_seed,
            git_commit=_current_git_commit(),
        )

    async def complete(self, experiment_id: str, metrics: dict[str, float]) -> Experiment:
        return await self._tracker.complete(experiment_id, metrics)

    async def fail(self, experiment_id: str, error: str) -> Experiment:
        return await self._tracker.fail(experiment_id, error)

    async def get(self, experiment_id: str) -> Experiment | None:
        return await self._tracker.get(experiment_id)

    async def list(self) -> list[Experiment]:
        return await self._tracker.list()
