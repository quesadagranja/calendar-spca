#!/usr/bin/env python3
"""Fit the selected SPCA-TV baseline with the paper's cyclic calendar graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from calendar_spca import ToroidalCalendarGraph
from spca_tv_operator import build_operator


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
        x = np.asarray(source, dtype=np.float64)
    else:
        imputed = np.asarray(source[:, int(layout["imputed_column"])])
        rows = np.flatnonzero(imputed <= float(layout["max_imputed"]))
        expected_n = int(layout["expected_rows_after_filter"])
        if rows.size != expected_n:
            raise ValueError(f"Expected {expected_n} eligible rows, found {rows.size}.")
        start = int(layout["feature_start"])
        stop = start + int(layout["feature_count"])
        x = np.asarray(source[rows, start:stop], dtype=np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Prepared feature matrix contains non-finite values.")
    x -= x.mean(axis=0, keepdims=True)
    return x


def git_commit_for_module(module_file: str) -> str | None:
    location = Path(module_file).resolve()
    for directory in (location.parent, *location.parents):
        if (directory / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                return None
    return None


def common_metrics(
    x: np.ndarray,
    loadings: np.ndarray,
    shape: tuple[int, int, int],
    order: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    norms = np.linalg.norm(loadings, axis=0)
    active = norms > 1e-12
    b = loadings[:, active]
    if b.shape[1] == 0:
        explained_variance = 0.0
    else:
        u, singular, _ = np.linalg.svd(b, full_matrices=False)
        rank = int(np.sum(singular > max(singular[0], 1.0) * 1e-12))
        q = u[:, :rank]
        projected = x @ q
        explained_variance = float(
            np.sum(projected * projected) / max(float(np.sum(x * x)), 1e-30)
        )

    sparsity = np.mean(np.abs(loadings) <= tolerance, axis=0)
    graph = ToroidalCalendarGraph(shape, order=order)
    tv = graph.total_variation(loadings)
    l1 = np.sum(np.abs(loadings), axis=0)
    rtv = np.divide(tv, l1, out=np.zeros_like(tv), where=l1 > 0)

    return {
        "effective_rank": int(np.count_nonzero(active)),
        "explained_variance": explained_variance,
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
    parser.add_argument("--alpha", type=float, help="Override the selected global scale.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-commit-check", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = load_json(config_path)
    settings = config["spca_tv"]
    data_path = resolve_data_path(args.dataset, args.data, args.paths)

    import parsimony
    from parsimony.decomposition import PCAL1L2TV

    required_commit = str(settings["commit"])
    installed_commit = git_commit_for_module(parsimony.__file__)
    if not args.skip_commit_check and installed_commit != required_commit:
        raise RuntimeError(
            "SPCA-TV reproduction requires ParsimonY commit "
            f"{required_commit}; detected {installed_commit!r}. "
            "See experiments/baselines/README.md for environment details."
        )

    x = prepare_data(data_path, config)
    shape = tuple(int(value) for value in config["calendar_shape"])
    order = str(config["order"])
    operator = build_operator(shape, order=order)

    alpha = float(args.alpha if args.alpha is not None else settings["selected_alpha"])
    base = settings["base_fractions_of_l1_max"]
    l1_max = float(PCAL1L2TV.l1_max(x))
    l1 = alpha * float(base["l1"]) * l1_max
    l2 = alpha * float(base["l2"]) * l1_max
    ltv = alpha * float(base["ltv"]) * l1_max

    print(f"Dataset:          {args.dataset}")
    print(f"Shape:            {x.shape}")
    print(f"Calendar:         {shape}, order={order}")
    print(f"ParsimonY commit: {installed_commit}")
    print(f"alpha:            {alpha:g}")
    print(f"l1_max:           {l1_max:.12g}")
    print(f"l1/l2/ltv:        {l1:.12g} / {l2:.12g} / {ltv:.12g}")
    if args.check_only:
        return

    random_state = int(settings["random_state"])
    np.random.seed(random_state)
    model = PCAL1L2TV(
        l1=l1,
        l2=l2,
        ltv=ltv,
        Atv=operator,
        n_components=int(config["n_components"]),
        criterion="frobenius",
        eps=float(settings["eps"]),
        max_iter=int(settings["max_iter"]),
        inner_eps=float(settings["inner_eps"]),
        inner_max_iter=int(settings["inner_max_iter"]),
        tau=float(settings["tau"]),
        verbose=True,
    )

    started = time.perf_counter()
    model.fit(x)
    elapsed = time.perf_counter() - started

    u = np.asarray(model.U, dtype=np.float64)
    v = np.asarray(model.V, dtype=np.float64)
    d = np.asarray(model.d, dtype=np.float64)
    effective_loadings = v * d[None, :]
    metrics = common_metrics(x, effective_loadings, shape, order)
    metrics.update(
        {
            "dataset": args.dataset,
            "random_state": random_state,
            "alpha": alpha,
            "l1_max": l1_max,
            "l1": l1,
            "l2": l2,
            "ltv": ltv,
            "elapsed_seconds": float(elapsed),
            "parsimony_commit": installed_commit,
        }
    )

    output = args.output_dir or Path("results/baselines") / args.dataset / "spca_tv"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "model.npz",
        U=u,
        V=v,
        d=d,
        effective_loadings=effective_loadings,
    )

    print(f"EV:       {100 * metrics['explained_variance']:.4f}%")
    print(f"Sparsity: {100 * metrics['mean_sparsity']:.4f}%")
    print(f"RTV:      {metrics['mean_rtv']:.6g}")
    print(f"Output:   {output}")


if __name__ == "__main__":
    main()
