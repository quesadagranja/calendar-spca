#!/usr/bin/env python3
"""Prepare a row-major float64 matrix for the R SPCA implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def selected_rows(source: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    layout = config["input_layout"]
    if layout["kind"] == "feature_matrix":
        if source.shape != (int(layout["expected_rows"]), int(layout["feature_count"])):
            raise ValueError(f"Unexpected prepared matrix shape: {source.shape}")
        return np.arange(source.shape[0], dtype=np.int64)

    imputed = np.asarray(source[:, int(layout["imputed_column"])])
    rows = np.flatnonzero(imputed <= float(layout["max_imputed"]))
    expected = int(layout["expected_rows_after_filter"])
    if rows.size != expected:
        raise ValueError(f"Expected {expected} eligible rows, found {rows.size}.")
    return rows


def feature_block(source: np.ndarray, rows: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    layout = config["input_layout"]
    if layout["kind"] == "feature_matrix":
        return source[rows]
    start = int(layout["feature_start"])
    stop = start + int(layout["feature_count"])
    return source[rows, start:stop]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("goiener", "lcl"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, help="Optional deterministic subset size.")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = load_json(config_path)
    data_path = resolve_data_path(args.dataset, args.data, args.paths)
    source = np.load(data_path, mmap_mode="r")
    if source.ndim != 2:
        raise SystemExit("Prepared data must be a two-dimensional NumPy array.")

    rows = selected_rows(source, config)
    if args.sample_size is not None:
        if args.sample_size < 1 or args.sample_size > rows.size:
            raise SystemExit("--sample-size must lie between 1 and the number of eligible rows.")
        rng = np.random.default_rng(args.seed)
        rows = rng.permutation(rows)[: args.sample_size]

    p = int(config["input_layout"]["feature_count"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        for start in range(0, rows.size, args.batch_size):
            stop = min(start + args.batch_size, rows.size)
            block = np.asarray(feature_block(source, rows[start:stop], config), dtype="<f8")
            if not np.all(np.isfinite(block)):
                raise RuntimeError("Prepared feature matrix contains non-finite values.")
            block.tofile(handle)

    metadata = {
        "dataset": args.dataset,
        "n_samples": int(rows.size),
        "n_features": p,
        "dtype": "little-endian float64",
        "layout": "row-major",
        "sample_seed": int(args.seed) if args.sample_size is not None else None,
        "sample_size_requested": args.sample_size,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {rows.size:,} x {p:,} matrix to {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
