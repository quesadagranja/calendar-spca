#!/usr/bin/env python3
"""Fit the rank-matched PCA reference used in the Calendar-SPCA paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

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


def prepare_data(path: Path, config: dict[str, Any]) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError("Prepared data must be two-dimensional.")
    layout = config["input_layout"]

    if layout["kind"] == "feature_matrix":
        expected = (int(layout["expected_rows"]), int(layout["feature_count"]))
        if tuple(source.shape) != expected:
            raise ValueError(f"Expected matrix shape {expected}, found {source.shape}.")
        x = np.asarray(source, dtype=np.float32)
    else:
        imputed = np.asarray(source[:, int(layout["imputed_column"])])
        rows = np.flatnonzero(imputed <= float(layout["max_imputed"]))
        expected_n = int(layout["expected_rows_after_filter"])
        if rows.size != expected_n:
            raise ValueError(f"Expected {expected_n} eligible rows, found {rows.size}.")
        start = int(layout["feature_start"])
        stop = start + int(layout["feature_count"])
        x = np.asarray(source[rows, start:stop], dtype=np.float32)

    if not np.all(np.isfinite(x)):
        raise ValueError("Prepared feature matrix contains non-finite values.")
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("goiener", "lcl"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = load_json(config_path)
    data_path = resolve_data_path(args.dataset, args.data, args.paths)
    x = prepare_data(data_path, config)

    n_components = int(config["n_components"])
    random_state = 111
    print(f"Dataset:   {args.dataset}")
    print(f"Shape:     {x.shape}")
    print(f"Rank:      {n_components}")
    print(f"SVD solver: randomized, random_state={random_state}")

    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    started = time.perf_counter()
    model.fit(x)
    elapsed = time.perf_counter() - started

    loadings = np.asarray(model.components_.T, dtype=np.float64)
    explained_variance = float(np.sum(model.explained_variance_ratio_))
    graph = ToroidalCalendarGraph(
        tuple(int(value) for value in config["calendar_shape"]),
        order=str(config["order"]),
    )
    l1 = np.sum(np.abs(loadings), axis=0)
    tv = graph.total_variation(loadings)
    rtv = np.divide(tv, l1, out=np.zeros_like(tv), where=l1 > 0)

    metrics = {
        "dataset": args.dataset,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_components": n_components,
        "svd_solver": "randomized",
        "random_state": random_state,
        "explained_variance": explained_variance,
        "mean_rtv": float(np.mean(rtv)),
        "component_rtv": rtv.tolist(),
        "elapsed_seconds": float(elapsed),
    }

    output = args.output_dir or Path("results/baselines") / args.dataset / "pca"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.save(output / "loadings.npy", loadings)

    print(f"EV:     {100 * explained_variance:.6f}%")
    print(f"RTV:    {metrics['mean_rtv']:.9f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
