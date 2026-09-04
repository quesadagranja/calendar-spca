"""Implicit operators for a toroidal calendar graph.

The default flattening convention is NumPy/C order: ``(hour, day, week)``
with the week index varying fastest. Use ``order="F"`` when the input was
flattened using R/Fortran array order.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_CALENDAR_SHAPE = (24, 7, 52)


def _check_order(order: str) -> str:
    order = order.upper()
    if order not in {"C", "F"}:
        raise ValueError("order must be 'C' (NumPy) or 'F' (R/Fortran).")
    return order


def n_calendar_cells(shape: Iterable[int] = DEFAULT_CALENDAR_SHAPE) -> int:
    """Return the number of nodes in a calendar grid."""

    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or any(s < 2 for s in shape):
        raise ValueError("calendar_shape must contain three integers >= 2.")
    return int(np.prod(shape))


def reshape_calendar(
    values: ArrayLike,
    shape: tuple[int, int, int] = DEFAULT_CALENDAR_SHAPE,
    *,
    order: str = "C",
) -> NDArray[np.floating]:
    """Reshape one or several flattened calendar vectors.

    ``values`` may have shape ``(p,)`` or ``(p, K)``. The returned shape is
    ``(H, D, W)`` or ``(H, D, W, K)``, respectively.
    """

    order = _check_order(order)
    p = n_calendar_cells(shape)
    arr = np.asarray(values)
    if arr.ndim == 1:
        if arr.size != p:
            raise ValueError(f"Expected {p} entries, got {arr.size}.")
        return arr.reshape(shape, order=order)
    if arr.ndim == 2:
        if arr.shape[0] != p:
            raise ValueError(f"Expected first dimension {p}, got {arr.shape[0]}.")
        return np.stack(
            [arr[:, k].reshape(shape, order=order) for k in range(arr.shape[1])],
            axis=-1,
        )
    raise ValueError("values must have shape (p,) or (p, K).")


def flatten_calendar(tensor: ArrayLike, *, order: str = "C") -> NDArray[np.floating]:
    """Flatten ``(H,D,W)`` or ``(H,D,W,K)`` calendar tensors."""

    order = _check_order(order)
    arr = np.asarray(tensor)
    if arr.ndim == 3:
        return arr.reshape(-1, order=order)
    if arr.ndim == 4:
        return np.column_stack(
            [arr[..., k].reshape(-1, order=order) for k in range(arr.shape[-1])]
        )
    raise ValueError("tensor must have shape (H,D,W) or (H,D,W,K).")


class ToroidalCalendarGraph:
    """Incidence operator of ``C_H square C_D square C_W`` without a matrix."""

    def __init__(
        self,
        shape: tuple[int, int, int] = DEFAULT_CALENDAR_SHAPE,
        *,
        order: str = "C",
    ) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.p = n_calendar_cells(self.shape)
        self.order = _check_order(order)

    @property
    def n_edges(self) -> int:
        """Number of oriented edges (three outgoing edges per node)."""

        return 3 * self.p

    @property
    def norm_sq(self) -> float:
        """Exact squared spectral norm of the incidence operator."""

        maxima = [
            4.0 if size % 2 == 0 else 2.0 + 2.0 * np.cos(np.pi / size)
            for size in self.shape
        ]
        return float(sum(maxima))

    def diff(self, values: ArrayLike) -> NDArray[np.floating]:
        """Apply ``D`` and return an array with shape ``(3,p,K)``.

        The orientation is current node minus its positive cyclic neighbour.
        For a vector input, a singleton component dimension is retained.
        """

        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] != self.p:
            raise ValueError(f"values must have shape ({self.p},) or ({self.p}, K).")
        tensor = reshape_calendar(arr, self.shape, order=self.order)
        differences = [tensor - np.roll(tensor, -1, axis=axis) for axis in range(3)]
        return np.stack(
            [flatten_calendar(item, order=self.order) for item in differences], axis=0
        )

    def adjoint(self, dual: ArrayLike) -> NDArray[np.floating]:
        """Apply the exact adjoint ``D.T`` to ``(3,p,K)`` dual variables."""

        arr = np.asarray(dual)
        was_vector = arr.ndim == 2
        if was_vector:
            arr = arr[..., None]
        if arr.ndim != 3 or arr.shape[:2] != (3, self.p):
            raise ValueError(f"dual must have shape (3,{self.p}) or (3,{self.p},K).")

        result = np.zeros((*self.shape, arr.shape[-1]), dtype=arr.dtype)
        for axis in range(3):
            field = reshape_calendar(arr[axis], self.shape, order=self.order)
            result += field - np.roll(field, 1, axis=axis)
        flat = flatten_calendar(result, order=self.order)
        return flat[:, 0] if was_vector else flat

    def total_variation(self, values: ArrayLike) -> NDArray[np.floating]:
        """Return anisotropic toroidal TV for every supplied component."""

        return np.sum(np.abs(self.diff(values)), axis=(0, 1))

    def connected_active_regions(
        self,
        values: ArrayLike,
        *,
        threshold: float = 1e-10,
    ) -> list[list[int]]:
        """Sizes of cyclic 3-D connected nonzero regions per component."""

        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[:, None]
        tensors = reshape_calendar(np.abs(arr) > threshold, self.shape, order=self.order)
        output: list[list[int]] = []
        h_max, d_max, w_max = self.shape

        for k in range(tensors.shape[-1]):
            mask = tensors[..., k]
            visited = np.zeros(self.shape, dtype=bool)
            sizes: list[int] = []
            for start in zip(*np.nonzero(mask & ~visited)):
                if visited[start]:
                    continue
                queue = deque([start])
                visited[start] = True
                size = 0
                while queue:
                    h, d, w = queue.popleft()
                    size += 1
                    neighbours = (
                        ((h + 1) % h_max, d, w),
                        ((h - 1) % h_max, d, w),
                        (h, (d + 1) % d_max, w),
                        (h, (d - 1) % d_max, w),
                        (h, d, (w + 1) % w_max),
                        (h, d, (w - 1) % w_max),
                    )
                    for neighbour in neighbours:
                        if mask[neighbour] and not visited[neighbour]:
                            visited[neighbour] = True
                            queue.append(neighbour)
                sizes.append(size)
            output.append(sorted(sizes, reverse=True))
        return output

    def active_region_statistics(
        self,
        values: ArrayLike,
        *,
        threshold: float = 1e-10,
    ) -> list[list[dict[str, float | int]]]:
        """Return size and loading mass of every cyclic active region.

        Regions are connected components of ``abs(values) > threshold`` using
        the six neighbours of the toroidal 3-D calendar graph. Components are
        sorted by decreasing number of cells. ``l1_fraction`` and
        ``l2_energy_fraction`` are relative to the complete active support of
        the corresponding loading column.
        """

        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] != self.p:
            raise ValueError(f"values must have shape ({self.p},) or ({self.p}, K).")

        absolute = reshape_calendar(np.abs(arr), self.shape, order=self.order)
        active = absolute > threshold
        output: list[list[dict[str, float | int]]] = []
        h_max, d_max, w_max = self.shape

        for k in range(absolute.shape[-1]):
            magnitudes = absolute[..., k]
            mask = active[..., k]
            total_l1 = float(np.sum(magnitudes[mask]))
            total_l2_energy = float(np.sum(magnitudes[mask] ** 2))
            visited = np.zeros(self.shape, dtype=bool)
            regions: list[dict[str, float | int]] = []

            for start in zip(*np.nonzero(mask & ~visited)):
                if visited[start]:
                    continue
                queue = deque([start])
                visited[start] = True
                size = 0
                l1_mass = 0.0
                l2_energy = 0.0
                while queue:
                    h, d, w = queue.popleft()
                    value = float(magnitudes[h, d, w])
                    size += 1
                    l1_mass += value
                    l2_energy += value * value
                    neighbours = (
                        ((h + 1) % h_max, d, w),
                        ((h - 1) % h_max, d, w),
                        (h, (d + 1) % d_max, w),
                        (h, (d - 1) % d_max, w),
                        (h, d, (w + 1) % w_max),
                        (h, d, (w - 1) % w_max),
                    )
                    for neighbour in neighbours:
                        if mask[neighbour] and not visited[neighbour]:
                            visited[neighbour] = True
                            queue.append(neighbour)

                regions.append(
                    {
                        "size": size,
                        "l1_mass": l1_mass,
                        "l1_fraction": l1_mass / max(total_l1, 1e-30),
                        "l2_energy": l2_energy,
                        "l2_energy_fraction": l2_energy
                        / max(total_l2_energy, 1e-30),
                    }
                )

            regions.sort(key=lambda item: int(item["size"]), reverse=True)
            output.append(regions)
        return output
