#!/usr/bin/env python3
"""Fit the pre-specified rank-15 diagonal Calendar-SPCA L-curve."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from calendar_spca import CalendarSPCA


HERE = Path(__file__).resolve().parent
DEFAULT_PATHS = HERE.parent / "paths.json"


class IndexedFeatureMatrix:
    """Row-filtered and column-sliced view over a NumPy matrix."""

    def __init__(
        self,
        base: np.ndarray,
        row_indices: np.ndarray,
        feature_start: int,
        feature_count: int,
    ) -> None:
        self.base = base
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.feature_start = int(feature_start)
        self.feature_stop = self.feature_start + int(feature_count)
        self.shape = (self.row_indices.size, int(feature_count))

    def __getitem__(self, key):
        rows = self.row_indices[key]
        return self.base[rows, self.feature_start : self.feature_stop]


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
        f"Data path for {dataset!r} is required. Pass --data PATH or create "
        "experiments/paths.json from experiments/paths.example.json."
    )


def prepare_matrix(data_path: str, config: dict[str, Any]):
    base = np.load(data_path, mmap_mode="r")
    if base.ndim != 2:
        raise ValueError("Prepared data must be a two-dimensional NumPy array.")

    layout = config["input_layout"]
    kind = layout["kind"]
    if kind == "feature_matrix":
        expected_p = int(layout["feature_count"])
        if base.shape[1] != expected_p:
            raise ValueError(f"Expected {expected_p} features, found {base.shape[1]}.")
        expected_n = layout.get("expected_rows")
        if expected_n is not None and base.shape[0] != int(expected_n):
            raise ValueError(f"Expected {expected_n} rows, found {base.shape[0]}.")
        return base

    if kind == "metadata_plus_features":
        feature_start = int(layout["feature_start"])
        feature_count = int(layout["feature_count"])
        if base.shape[1] < feature_start + feature_count:
            raise ValueError("Prepared matrix does not contain the configured feature block.")
        imputed = np.asarray(base[:, int(layout["imputed_column"])])
        eligible = np.flatnonzero(imputed <= float(layout["max_imputed"]))
        expected = layout.get("expected_rows_after_filter")
        if expected is not None and eligible.size != int(expected):
            raise ValueError(
                f"Expected {expected} eligible rows, found {eligible.size}."
            )
        return IndexedFeatureMatrix(base, eligible, feature_start, feature_count)

    raise ValueError(f"Unknown input_layout.kind: {kind}")


def result_name(index: int, value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return f"lambda_{index:02d}_{text}.json"


def fit_lambda(
    index: int,
    lam: float,
    data_path: str,
    config: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    fits_dir = output / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    result_path = fits_dir / result_name(index, lam)
    if result_path.is_file():
        return {"lambda": lam, "status": "skipped"}

    matrix = prepare_matrix(data_path, config)
    model = CalendarSPCA(
        n_components=int(config["n_components"]),
        lambda_l1=float(lam),
        lambda_tv=float(lam),
        calendar_shape=tuple(int(x) for x in config["calendar_shape"]),
        order=str(config["order"]),
        center=bool(config["center"]),
        random_state=int(config["random_state"]),
        **dict(config["model"]),
    )

    started = time.perf_counter()
    model.fit(matrix)
    elapsed = time.perf_counter() - started

    reconstruction_ss = float(model.reconstruction_error_)
    residual_norm = float(np.sqrt(max(reconstruction_ss, 0.0)))
    loading_l1 = float(np.sum(model.loading_l1_norm_))
    graph_tv = float(np.sum(model.total_variation_))
    omega = loading_l1 + graph_tv
    active = np.flatnonzero(model.active_)

    result = {
        "lambda": float(lam),
        "converged": bool(model.converged_),
        "effective_rank": int(model.n_components_effective_),
        "outer_iterations": int(model.n_iter_),
        "explained_variance": float(model.explained_variance_),
        "reconstruction_ss": reconstruction_ss,
        "rho": residual_norm,
        "loading_l1": loading_l1,
        "graph_tv": graph_tv,
        "omega": omega,
        "mean_sparsity_active": (
            float(np.mean(model.loading_sparsity_[active])) if active.size else float("nan")
        ),
        "mean_relative_tv_active": (
            float(np.mean(model.relative_total_variation_[active]))
            if active.size
            else float("nan")
        ),
        "runtime_seconds": float(elapsed),
    }

    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    tmp.replace(result_path)
    return {"lambda": lam, "status": "completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("goiener", "lcl"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-points", type=int, help="Fit only the first N path points.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = load_json(config_path)
    data_path = resolve_data_path(args.dataset, args.data, args.paths)
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    # Validate the complete input once before starting expensive fits.
    matrix = prepare_matrix(str(data_path), config)
    output_dir = args.output_dir or Path("results/model_selection") / args.dataset
    points = list(enumerate(float(x) for x in config["lambda_path"]))
    if args.max_points is not None:
        points = points[: max(0, args.max_points)]

    print(f"Dataset:   {args.dataset}")
    print(f"Data:      {data_path}")
    print(f"Shape:     {matrix.shape}")
    print(f"Rank:      {config['n_components']}")
    print(f"Path size: {len(points)}")
    print(f"Output:    {output_dir}")
    print("Warm starts: disabled (each point is fitted independently)")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.snapshot.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    worker_args = [
        (index, lam, str(data_path), config, str(output_dir))
        for index, lam in points
    ]

    failed = 0
    if args.workers == 1:
        for position, item in enumerate(worker_args, start=1):
            result = fit_lambda(*item)
            print(
                f"[{position:02d}/{len(points):02d}] {result['status']:9s} "
                f"lambda={result['lambda']:g}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fit_lambda, *item): item for item in worker_args}
            for position, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    print(
                        f"[{position:02d}/{len(points):02d}] FAILED lambda={item[1]:g}: {exc}",
                        flush=True,
                    )
                    continue
                print(
                    f"[{position:02d}/{len(points):02d}] {result['status']:9s} "
                    f"lambda={result['lambda']:g}",
                    flush=True,
                )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
