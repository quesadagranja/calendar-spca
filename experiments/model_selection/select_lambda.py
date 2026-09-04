#!/usr/bin/env python3
"""Select Calendar-SPCA regularization by discrete L-curve curvature."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    files = sorted((output_dir / "fits").glob("lambda_*.json"))
    if not files:
        raise SystemExit(f"No L-curve fit files found under {output_dir / 'fits'}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    denominator = ab * bc * ca
    if denominator <= 0:
        return float("nan")
    twice_area = abs(float(np.cross(b - a, c - a)))
    return 2.0 * twice_area / denominator


def select(rows: list[dict[str, Any]], rank: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        rho = float(row.get("rho", float("nan")))
        omega = float(row.get("omega", float("nan")))
        row["selection_candidate"] = bool(
            row.get("converged")
            and int(row.get("effective_rank", -1)) == rank
            and math.isfinite(rho)
            and math.isfinite(omega)
            and rho > 0
            and omega > 0
        )
        row["lcurve_curvature"] = float("nan")
        row["selected"] = False
        enriched.append(row)

    candidates = sorted(
        [row for row in enriched if row["selection_candidate"]],
        key=lambda row: float(row["lambda"]),
    )
    if len(candidates) < 3:
        raise RuntimeError(
            "At least three converged full-rank candidates are required for L-curve selection."
        )

    for index in range(1, len(candidates) - 1):
        triple = candidates[index - 1 : index + 2]
        points = [
            np.array(
                [math.log10(float(row["omega"])), math.log10(float(row["rho"]))],
                dtype=np.float64,
            )
            for row in triple
        ]
        candidates[index]["lcurve_curvature"] = menger_curvature(*points)

    interior = [
        row
        for row in candidates[1:-1]
        if math.isfinite(float(row["lcurve_curvature"]))
    ]
    if not interior:
        raise RuntimeError("No finite interior L-curve curvature could be computed.")

    selected = sorted(
        interior,
        key=lambda row: (-float(row["lcurve_curvature"]), float(row["lambda"])),
    )[0]
    selected["selected"] = True
    selected_index = candidates.index(selected)
    neighbourhood = candidates[selected_index - 1 : selected_index + 2]

    selection = {
        "lambda_star": float(selected["lambda"]),
        "curvature": float(selected["lcurve_curvature"]),
        "n_candidates": len(candidates),
        "candidate_lambdas": [float(row["lambda"]) for row in candidates],
        "curvature_neighborhood_lambdas": [
            float(row["lambda"]) for row in neighbourhood
        ],
        "method": "maximum discrete Menger curvature on (log10 omega, log10 rho)",
        "candidate_rule": (
            f"converged, effective_rank={rank}, and finite positive rho/omega"
        ),
        "tie_break": "smaller lambda for an exact curvature tie",
    }
    return selection, enriched


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "lambda",
        "converged",
        "effective_rank",
        "outer_iterations",
        "explained_variance",
        "reconstruction_ss",
        "rho",
        "loading_l1",
        "graph_tv",
        "omega",
        "mean_sparsity_active",
        "mean_relative_tv_active",
        "runtime_seconds",
        "selection_candidate",
        "lcurve_curvature",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["lambda"])):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("goiener", "lcl"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or HERE / "config" / f"{args.dataset}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path("results/model_selection") / args.dataset

    rows = load_results(output_dir)
    selection, enriched = select(rows, int(config["n_components"]))
    write_table(enriched, output_dir / "lcurve.csv")
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Dataset:      {args.dataset}")
    print(f"Candidates:   {selection['n_candidates']}")
    print(f"Selected λ*:  {selection['lambda_star']:g}")
    print(f"Curvature:    {selection['curvature']:.8g}")
    print(f"Neighbourhood:{selection['curvature_neighborhood_lambdas']}")
    print(f"Wrote:        {output_dir / 'lcurve.csv'}")
    print(f"Wrote:        {output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
