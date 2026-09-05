"""Component matching and multi-start stability utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from .api import CalendarSPCA


@dataclass
class ComponentMatch:
    reference_indices: NDArray[np.int_]
    candidate_indices: NDArray[np.int_]
    absolute_cosines: NDArray[np.float64]
    mean_similarity: float


def match_components(reference: ArrayLike, candidate: ArrayLike) -> ComponentMatch:
    """Match loading columns by maximum total absolute cosine similarity."""

    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Both inputs must have shape (p, K), with the same p.")
    denom = np.maximum(
        np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :],
        1e-30,
    )
    similarities = np.abs(a.T @ b) / denom
    rows, cols = linear_sum_assignment(-similarities)
    values = similarities[rows, cols]
    return ComponentMatch(
        rows,
        cols,
        values,
        float(np.mean(values)) if values.size else 0.0,
    )


def fit_restarts(
    estimator: CalendarSPCA,
    x: ArrayLike,
    *,
    n_restarts: int = 5,
    seeds: Sequence[int] | None = None,
) -> tuple[
    CalendarSPCA,
    list[CalendarSPCA],
    NDArray[np.float64],
]:
    """Fit several initializations and return best model plus stability matrix."""

    if n_restarts < 1:
        raise ValueError("n_restarts must be positive.")
    if seeds is None:
        base = 0 if estimator.random_state is None else estimator.random_state
        seeds = [base + i for i in range(n_restarts)]
    if len(seeds) != n_restarts:
        raise ValueError("seeds must have length n_restarts.")

    models: list[CalendarSPCA] = []
    for seed in seeds:
        parameters = estimator.get_params()
        parameters["random_state"] = int(seed)
        model = CalendarSPCA(**parameters)
        models.append(model.fit(x))

    final_objectives = np.array([model.history_[-1].objective for model in models])
    best = models[int(np.argmin(final_objectives))]
    stability = np.eye(n_restarts, dtype=np.float64)
    for i in range(n_restarts):
        for j in range(i + 1, n_restarts):
            ai = models[i].components_[:, models[i].active_]
            bj = models[j].components_[:, models[j].active_]
            value = match_components(ai, bj).mean_similarity
            stability[i, j] = stability[j, i] = value
    return best, models, stability
