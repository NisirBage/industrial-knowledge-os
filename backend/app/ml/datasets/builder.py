"""Training Dataset Builder (M6 Component 1): turns labeled equipment ids
into a supervised pandas DataFrame via the Feature Store — no model-specific
logic lives here, only feature assembly and standard sklearn splitting
utilities. Every one of the seven MLTaskType tasks uses this same builder;
what differs per task is only which labels the caller supplies.
"""

import dataclasses
from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

from app.features.models import EquipmentFeatureVector


class FeatureExtractorProtocol(Protocol):
    async def extract(self, equipment_id: str) -> EquipmentFeatureVector: ...


class TrainingDatasetBuilder:
    def __init__(self, feature_extractor: FeatureExtractorProtocol):
        self._feature_extractor = feature_extractor

    async def build(self, labeled_equipment: dict[str, float], label_column: str = "label") -> pd.DataFrame:
        rows = []
        for equipment_id, label in labeled_equipment.items():
            features = await self._feature_extractor.extract(equipment_id)
            row = dataclasses.asdict(features)
            row[label_column] = label
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def train_test_split(
        df: pd.DataFrame, label_column: str = "label", test_size: float = 0.2, random_state: int = 42, stratify: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stratify_values = df[label_column] if stratify else None
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_values)

    @staticmethod
    def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> list[tuple]:
        """Uses row order as the timeline proxy — real historical data would
        supply an actual timestamp column to sort by first; documented
        limitation until ingestion captures per-record dates (see
        TimelineEvent.timestamp's own docstring for the same gap)."""
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return list(splitter.split(df))

    @staticmethod
    def cross_validation_folds(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> list[tuple]:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(df))

    @staticmethod
    def to_csv(df: pd.DataFrame, path: str | Path) -> None:
        df.to_csv(path, index=False)

    @staticmethod
    def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
        df.to_parquet(path, index=False)
