#!/usr/bin/env python3
"""Evaluate a loading matrix with the common paper metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from calendar_spca import ToroidalCalendarGraph


HERE = Path(__file__).resolve().parent
DEFAULT_PATHS = HERE.parent / "paths.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_data_path(dataset: str, cli_path: Path | None, paths_file: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    candidate = paths_file or DEFAULT_PATHS
    if candidate.is_file():
        paths = load_json(candidate)
        if dataset in paths:
            return Path(paths[dataset]).expanduser().resolve()
    raise SystemExit(
        f"Data path for {dataset!r} is required. Pass --data PATH or create experiments/paths.json."
    )


class PreparedData:
    def __init__(self, path: Path, config: dict[str, Any], batch_size: int) -> None:
        self.source = np.load(path, mmap_mode="r")
        self.config = config
        self.batch_size = int(batch_size)
        layout = config["input_layout"]
        self.p = int(layout["feature_count"])

        if layout["kind"] == "feature_matrix":
            expected = (int(layout["expected_rows"]), self.p)
            if tuple(self.source.shape) != expected:
                raise ValueError(f"Expected matrix shape {expected}, found {self.source.shape}.")
            self.rows = np.arange(self.source.shape[0], dtype=np.int64)
            self.feature_start = 0
        else:
            imputed = np.asarray(self.source[:, int(layout["imputed_column"])])
            self.rows = np.flatnonzero(imputed <= float(layout["max_imputed"]))
            expected_n = int(layout["expected_rows_after_filter"])
            if self.rows.size != expected_n:
                raise ValueError(f"Expected {expected_n} eligible rows, found {self.rows.size}.")
            self.feature_start = int(layout["feature_start"])
        self.n = int(self.rows.size)

    def blocks(self) -> Iterator[np.ndarray]:
        stop_feature = self.feature_start + self.p
        for start in range(0, self.n, self.batch_size):
            stop = min(start + self.batch_size, self.n)
            rows = self.rows[start:stop]
            block = np.asarray(
                self.source[rows, self.feature_start:stop_feature], dtype=np.float64
            )
            if not np.all(np.isfinite(block)):
                raise ValueError("Prepared data contain non-finite values.")
            yield block


def load_loadings(path: Path, kind: str, p: int, k: int, key: str) -> np.ndarray:
    if kind == "raw_colmajor":
        values = np.fromfile(path, dtype="<f8")
        if values.size != p * k:
            raise ValueError(f"Expected {p*k} loading values, found {values.size}.")
        return values.reshape((p, k), order="F")
    if kind == "npy":
        loadings = np.asarray(np.load(path), dtype=np.float64)
    elif kind == "npz":
        with np.load(path) as archive:
            if key not in archive:
                raise ValueError(f"Key {key!r} not found in {path}.")
            loadings = np.asarray(archive[key], dtype=np.float64)
    else:
        raise ValueError(f"Unknown loading format: {kind}")
    if loadings.shape != (p, k):
        raise ValueError(f"Expected loading shape {(p, k)}, found {loadings.shape}.")
    return loadings


def evaluate(
    data: PreparedData,
    loadings: np.ndarray,
    shape: tuple[int, int, int],
    order: str,
    tolerance: float,
) -> dict[str, Any]:
    mean_vector = np.zeros(data.p, dtype=np.float64)
    for block in data.blocks():
        mean_vector += block.sum(axis=0)
    mean_vector /= data.n

    norms = np.linalg.norm(loadings, axis=0)
    active = norms > 1e-12
    b = loadings[:, active]
    if b.shape[1]:
        u, singular, _ = np.linalg.svd(b, full_matrices=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-12))
        q = u[:, :rank]
    else:
        rank = 0
        q = np.empty((data.p, 0), dtype=np.float64)

    total_ss = 0.0
    projected_ss = 0.0
    for block in data.blocks():
        centered = block - mean_vector
        total_ss += float(np.einsum("ij,ij->", centered, centered))
        if rank:
            scores = centered @ q
            projected_ss += float(np.einsum("ij,ij->", scores, scores))

    graph = ToroidalCalendarGraph(shape, order=order)
    sparsity = np.mean(np.abs(loadings) <= tolerance, axis=0)
    l1 = np.sum(np.abs(loadings), axis=0)
    tv = graph.total_variation(loadings)
    rtv = np.divide(tv, l1, out=np.zeros_like(tv), where=l1 > 0)

    return {
        "n_samples": data.n,
        "n_features": data.p,
        "n_components": int(loadings.shape[1]),
        "effective_rank": int(np.count_nonzero(active)),
        "loading_rank": rank,
        "explained_variance": projected_ss / max(total_ss, 1e-30),
        "mean_sparsity": float(np.mean(sparsity[active])) if np.any(active) else float("nan"),
        "mean_rtv": float(np.mean(rtv[active])) if np.any(active) else float("nan"),
        "component_sparsity": sparsity.tolist(),
        "component_rtv": rtv.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("goiener", "lcl"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--loadings", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("raw_colmajor", "npy", "npz"),
        default="raw_colmajor",
    )
    parser.add_argument("--key", default="effective_loadings")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sparsity-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = load_json(config_path)
    data_path = resolve_data_path(args.dataset, args.data, args.paths)
    data = PreparedData(data_path, config, args.batch_size)
    k = int(config["n_components"])
    loadings = load_loadings(args.loadings, args.format, data.p, k, args.key)
    metrics = evaluate(
        data,
        loadings,
        tuple(int(value) for value in config["calendar_shape"]),
        str(config["order"]),
        float(args.sparsity_tolerance),
    )

    text = json.dumps(metrics, indent=2, allow_nan=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
