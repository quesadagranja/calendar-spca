#!/usr/bin/env python3
"""Aggregate factorial fit outputs into paper-level summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from calendar_spca import match_components


METRICS = (
    "explained_variance",
    "mean_sparsity_active",
    "mean_relative_tv_active",
    "mean_effective_regions_active",
    "runtime_seconds",
)


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    files = sorted((output_dir / "fits").glob("*.json"))
    if not files:
        raise SystemExit(f"No fit JSON files found under {output_dir / 'fits'}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def write_fit_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task",
        "repeat",
        "sample_seed",
        "initialization_seed",
        "n_samples",
        "n_components",
        "lambda_l1",
        "lambda_tv",
        "converged",
        "effective_rank",
        "outer_iterations",
        "explained_variance",
        "mean_sparsity_active",
        "mean_relative_tv_active",
        "mean_effective_regions_active",
        "condition_number_v_gram",
        "final_objective",
        "runtime_seconds",
        "components_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                r["n_samples"],
                r["n_components"],
                r["lambda_l1"],
                r["lambda_tv"],
                r["repeat"],
            ),
        ):
            writer.writerow(row)


def stability_for_group(group: list[dict[str, Any]], output_dir: Path) -> float:
    eligible = [
        row
        for row in group
        if row.get("converged")
        and int(row.get("effective_rank", -1)) == int(row["n_components"])
        and row.get("components_file")
    ]
    if len(eligible) < 2:
        return float("nan")

    similarities: list[float] = []
    for left, right in combinations(eligible, 2):
        a_file = output_dir / left["components_file"]
        b_file = output_dir / right["components_file"]
        if not (a_file.is_file() and b_file.is_file()):
            continue
        with np.load(a_file) as a_npz, np.load(b_file) as b_npz:
            a = np.asarray(a_npz["components"])
            b = np.asarray(b_npz["components"])
            a_active = np.asarray(a_npz["active"], dtype=bool)
            b_active = np.asarray(b_npz["active"], dtype=bool)
        similarities.append(match_components(a[:, a_active], b[:, b_active]).mean_similarity)
    return float(np.mean(similarities)) if similarities else float("nan")


def summarize(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["n_samples"]),
            int(row["n_components"]),
            float(row["lambda_l1"]),
            float(row["lambda_tv"]),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        n_samples, n_components, lambda_l1, lambda_tv = key
        converged = [row for row in group if row.get("converged")]
        full_rank = [
            row
            for row in converged
            if int(row.get("effective_rank", -1)) == n_components
        ]
        summary: dict[str, Any] = {
            "n_samples": n_samples,
            "n_components": n_components,
            "lambda_l1": lambda_l1,
            "lambda_tv": lambda_tv,
            "n_fits": len(group),
            "n_converged": len(converged),
            "n_converged_full_rank": len(full_rank),
            "reliable_4_of_5": len(converged) >= 4,
            "mean_pairwise_component_similarity": stability_for_group(group, output_dir),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in converged if np.isfinite(float(row[metric]))]
            summary[f"mean_{metric}"] = mean(values) if values else float("nan")
            summary[f"sd_{metric}"] = pstdev(values) if len(values) > 1 else 0.0 if values else float("nan")
        summaries.append(summary)
    return summaries


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/factorial"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_fit_table(rows, args.output_dir / "fits.csv")
    summary = summarize(rows, args.output_dir)
    write_summary(summary, args.output_dir / "factorial_summary.csv")
    print(f"Fits:          {len(rows):,}")
    print(f"Configurations:{len(summary):,}")
    print(f"Wrote:         {args.output_dir / 'fits.csv'}")
    print(f"Wrote:         {args.output_dir / 'factorial_summary.csv'}")


if __name__ == "__main__":
    main()
