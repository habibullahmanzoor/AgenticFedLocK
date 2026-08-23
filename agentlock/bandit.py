from __future__ import annotations

import numpy as np


class LinUCBBandit:
    def __init__(self, arms: tuple[str, ...], feature_dim: int, alpha: float = 0.4, ridge_lambda: float = 1.0) -> None:
        self.arms = arms
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self._matrices = {arm: np.eye(feature_dim, dtype=np.float64) * ridge_lambda for arm in arms}
        self._vectors = {arm: np.zeros(feature_dim, dtype=np.float64) for arm in arms}
        self._counts = {arm: 0 for arm in arms}

    def score(self, arm: str, context: np.ndarray) -> dict[str, float]:
        matrix = self._matrices[arm]
        vector = self._vectors[arm]
        matrix_inv = np.linalg.inv(matrix)
        theta = matrix_inv @ vector
        mean = float(context @ theta)
        uncertainty = float(self.alpha * np.sqrt(np.clip(context @ matrix_inv @ context, 0.0, None)))
        return {
            "mean": mean,
            "optimistic": mean + uncertainty,
            "uncertainty": uncertainty,
        }

    def scores(self, context: np.ndarray) -> dict[str, dict[str, float]]:
        return {arm: self.score(arm, context) for arm in self.arms}

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        self._matrices[arm] += np.outer(context, context)
        self._vectors[arm] += reward * context
        self._counts[arm] += 1

    def update_batch(self, feedback_rows: list[dict[str, object]]) -> None:
        for row in feedback_rows:
            context = np.asarray(row["context_vector"], dtype=np.float64)
            self.update(str(row["template"]), context, float(row["reward"]))

    def count(self, arm: str) -> int:
        return self._counts[arm]

    def min_count(self) -> int:
        return min(self._counts.values())

    def counts(self) -> dict[str, int]:
        return dict(self._counts)
