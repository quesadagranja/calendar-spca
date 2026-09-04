#!/usr/bin/env python3
"""Run the Calendar-SPCA factorial experiment from the paper.

The scientific design is stored in ``config.json``. Dataset and output paths
are supplied at runtime, keeping machine-specific locations out of version
control.
"""

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
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_PATHS = HERE.parent / "paths.json"


class IndexedFeatureMatrix:
    """Row-indexed, column-sliced view over an on-disk NumPy matrix."""

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
        selected = self.row_indices[key]
        return self.base[selected, self.feature_start : self.feature_stop]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_data_path(cli_path: Path | None, paths_file: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    candidate = paths_file or DEFAULT_PATHS
    if candidate.is_file():
        paths = load_json(candidate)
        if "goiener" in paths:
            return Path(paths["goiener"]).expanduser().resolve()
    raise SystemExit(
        "GoiEner data path is required. Pass --data PATH or create "
        "experiments/paths.json from experiments/paths.example.json."
    )


def task_name(task: dict[str, Any]) -> str:
    return (
        f"r{task['repeat']:02d}_n{task['n_samples']}_k{task['n_components']}_"
        f"l1{task['lambda_l1_index']:02d}_tv{task['lambda_tv_index']:02d}"
    )


def eligible_indices(base: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    layout = config["input_layout"]
    imputed = np.asarray(base[:, int(layout["imputed_column"])])
    return np.flatnonzero(imputed <= float(layout["max_imputed"]))


def sampled_indices(
    eligible: np.ndarray,
    sample_seed: int,
    n_samples: int,
    max_n: int,
) -> np.ndarray:
    """Regenerate the nested master sample for one repetition."""

    rng = np.random.default_rng(int(sample_seed))
    master = rng.permutation(eligible)[:max_n]
    return np.asarray(master[:n_samples], dtype=np.int64)


def build_tasks(config: dict[str, Any], n_eligible: int) -> list[dict[str, Any]]:
    max_n = max(int(n) for n in config["sample_sizes"])
    if n_eligible < max_n:
        raise ValueError(
            f"Only {n_eligible:,} eligible rows are available; {max_n:,} are required."
        )

    tasks: list[dict[str, Any]] = []
    for repeat, (sample_seed, initialization_base) in enumerate(
        zip(config["sample_seeds"], config["initialization_base_seeds"], strict=True),
        start=1,
    ):
        for n_index, n_samples in enumerate(config["sample_sizes"]):
            n_samples = int(n_samples)
            for k_index, n_components in enumerate(config["n_components_values"]):
                n_components = int(n_components)
                initialization_seed = int(initialization_base) + 1000 * n_index + k_index

                for l1_index, lambda_l1 in enumerate(config["lambda_l1_values"]):
                    for tv_index, lambda_tv in enumerate(config["lambda_tv_values"]):
                        tasks.append(
                            {
                                "repeat": repeat,
                                "sample_seed": int(sample_seed),
                                "initialization_seed": initialization_seed,
                                "n_samples": n_samples,
                                "n_components": n_components,
                                "lambda_l1": float(lambda_l1),
                                "lambda_tv": float(lambda_tv),
                                "lambda_l1_index": l1_index,
                                "lambda_tv_index": tv_index,
                            }
                        )
    return tasks


def fit_one(
    task: dict[str, Any],
    data_path: str,
    config: dict[str, Any],
    output_dir: str,
    save_components: bool,
) -> dict[str, Any]:
    output = Path(output_dir)
    fits_dir = output / "fits"
    components_dir = output / "components"
    fits_dir.mkdir(parents=True, exist_ok=True)
    if save_components:
        components_dir.mkdir(parents=True, exist_ok=True)

    name = task_name(task)
    result_path = fits_dir / f"{name}.json"
    if result_path.is_file():
        return {"task": name, "status": "skipped"}

    base = np.load(data_path, mmap_mode="r")
    layout = config["input_layout"]
    eligible = eligible_indices(base, config)
    max_n = max(int(n) for n in config["sample_sizes"])
    rows = sampled_indices(
        eligible,
        int(task["sample_seed"]),
        int(task["n_samples"]),
        max_n,
    )
    matrix = IndexedFeatureMatrix(
        base,
        rows,
        int(layout["feature_start"]),
        int(layout["feature_count"]),
    )

    model_kwargs = dict(config["model"])
    model = CalendarSPCA(
        n_components=int(task["n_components"]),
        lambda_l1=float(task["lambda_l1"]),
        lambda_tv=float(task["lambda_tv"]),
        calendar_shape=tuple(int(x) for x in config["calendar_shape"]),
        order=str(config["order"]),
        center=bool(config["center"]),
        random_state=int(task["initialization_seed"]),
        **model_kwargs,
    )

    started = time.perf_counter()
    model.fit(matrix)
    elapsed = time.perf_counter() - started

    active = np.flatnonzero(model.active_)
    mean_rtv = (
        float(np.mean(model.relative_total_variation_[active]))
        if active.size
        else float("nan")
    )

    result = {
        "task": name,
        "repeat": int(task["repeat"]),
        "sample_seed": int(task["sample_seed"]),
        "initialization_seed": int(task["initialization_seed"]),
        "n_samples": int(task["n_samples"]),
        "n_components": int(task["n_components"]),
        "lambda_l1": float(task["lambda_l1"]),
        "lambda_tv": float(task["lambda_tv"]),
        "converged": bool(model.converged_),
        "effective_rank": int(model.n_components_effective_),
        "outer_iterations": int(model.n_iter_),
        "explained_variance": float(model.explained_variance_),
        "mean_sparsity_active": float(model.mean_loading_sparsity_active_),
        "mean_relative_tv_active": mean_rtv,
        "mean_effective_regions_active": float(model.mean_effective_regions_active_),
        "condition_number_v_gram": float(model.condition_number_),
        "final_objective": float(model.history_[-1].objective),
        "runtime_seconds": float(elapsed),
    }

    if save_components:
        component_name = f"{name}.npz"
        np.savez_compressed(
            components_dir / component_name,
            components=model.components_,
            active=model.active_,
        )
        result["components_file"] = f"components/{component_name}"

    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    tmp.replace(result_path)
    return {"task": name, "status": "completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Prepared GoiEner .npy matrix.")
    parser.add_argument("--paths", type=Path, help="Optional local paths.json file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("results/factorial"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Store fitted loadings for repeat-stability analysis.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Run only the first N tasks (useful for smoke tests).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    config = load_json(args.config)
    data_path = resolve_data_path(args.data, args.paths)
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    base = np.load(data_path, mmap_mode="r")
    if base.ndim != 2:
        raise SystemExit("Prepared data must be a two-dimensional NumPy array.")

    layout = config["input_layout"]
    feature_stop = int(layout["feature_start"]) + int(layout["feature_count"])
    if base.shape[1] < feature_stop:
        raise SystemExit(
            f"Prepared matrix has {base.shape[1]} columns; at least {feature_stop} are required."
        )

    eligible = eligible_indices(base, config)
    tasks = build_tasks(config, eligible.size)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]

    print(f"Data:          {data_path}")
    print(f"Shape:         {base.shape}")
    print(f"Eligible rows: {eligible.size:,}")
    print(f"Tasks:         {len(tasks):,}")
    print(f"Workers:       {args.workers}")
    print(f"Output:        {args.output_dir}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.output_dir / "config.snapshot.json"
    snapshot.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    worker_args = [
        (task, str(data_path), config, str(args.output_dir), args.save_components)
        for task in tasks
    ]

    completed = skipped = failed = 0
    if args.workers == 1:
        iterator = (fit_one(*item) for item in worker_args)
        for index, result in enumerate(iterator, start=1):
            if result["status"] == "completed":
                completed += 1
            else:
                skipped += 1
            print(
                f"[{index:>5}/{len(tasks)}] {result['status']:9s} {result['task']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fit_one, *item): item[0] for item in worker_args}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"[{index:>5}/{len(tasks)}] FAILED    {task_name(task)}: {exc}", flush=True)
                    continue
                if result["status"] == "completed":
                    completed += 1
                else:
                    skipped += 1
                print(
                    f"[{index:>5}/{len(tasks)}] {result['status']:9s} {result['task']}",
                    flush=True,
                )

    print(f"Completed: {completed:,}; skipped: {skipped:,}; failed: {failed:,}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
