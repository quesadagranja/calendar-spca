"""Cyclic calendar operator for the ParsimonY SPCA-TV baseline."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from parsimony.utils.linalgs import LinearOperatorNesterov


def _cycle_laplacian_lambda_max(size: int) -> float:
    if size < 2:
        raise ValueError("Every cyclic axis must contain at least two positions.")
    if size % 2 == 0:
        return 4.0
    return 4.0 * math.cos(math.pi / (2.0 * size)) ** 2


def cyclic_calendar_lambda_max(shape: tuple[int, int, int]) -> float:
    return float(sum(_cycle_laplacian_lambda_max(int(size)) for size in shape))


def build_operator(
    shape: tuple[int, int, int],
    *,
    order: str = "F",
) -> LinearOperatorNesterov:
    """Return the three-direction cyclic operator expected by PCAL1L2TV."""

    if order not in {"C", "F"}:
        raise ValueError("order must be 'C' or 'F'")
    h, d, w = (int(value) for value in shape)
    p = h * d * w

    def index(ih: int, iday: int, iw: int) -> int:
        return int(np.ravel_multi_index((ih, iday, iw), shape, order=order))

    matrices = []
    for axis in range(3):
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for iw in range(w):
            for iday in range(d):
                for ih in range(h):
                    current = [ih, iday, iw]
                    neighbour = current.copy()
                    limits = (h, d, w)
                    neighbour[axis] = (neighbour[axis] + 1) % limits[axis]
                    i = index(*current)
                    j = index(*neighbour)
                    rows.extend((i, i))
                    cols.extend((i, j))
                    values.extend((-1.0, 1.0))
        matrices.append(
            sp.csr_matrix((values, (rows, cols)), shape=(p, p))
        )

    operator = LinearOperatorNesterov(*matrices)
    operator.n_compacts = p
    operator.singular_values = [cyclic_calendar_lambda_max(shape)]
    return operator
