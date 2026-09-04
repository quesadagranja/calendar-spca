"""Calendar-Structured Graph-Fused Sparse PCA estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .graph import DEFAULT_CALENDAR_SHAPE, ToroidalCalendarGraph, reshape_calendar


FloatArray = NDArray[np.float64]


def _soft_threshold(x: FloatArray, threshold: float) -> FloatArray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _safe_relative(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / (denominator + eps))


class _CenteredMatrix:
    """Read-only centered linear operator backed by an array or memmap."""

    def __init__(self, x: ArrayLike, center: bool, batch_size: int) -> None:
        if not hasattr(x, "shape"):
            x = np.asarray(x)
        if len(x.shape) != 2:  # type: ignore[arg-type]
            raise ValueError("X must be a two-dimensional numeric array.")
        if not np.issubdtype(np.asarray(x[:1]).dtype, np.number):
            raise TypeError("X must contain numeric values only.")
        self.x = x
        self.n, self.p = map(int, x.shape)  # type: ignore[union-attr]
        self.batch_size = max(1, int(batch_size))
        self.mean = np.zeros(self.p, dtype=np.float64)
        column_sum_sq = np.zeros(self.p, dtype=np.float64)

        if center:
            for start, stop in self._batches():
                block = np.asarray(self.x[start:stop], dtype=np.float64)
                if not np.all(np.isfinite(block)):
                    raise ValueError("X contains NaN or infinite values.")
                self.mean += np.sum(block, axis=0)
                column_sum_sq += np.einsum("ij,ij->j", block, block)
            self.mean /= self.n
            self.norm_sq = float(np.sum(column_sum_sq) - self.n * self.mean @ self.mean)
        else:
            total = 0.0
            for start, stop in self._batches():
                block = np.asarray(self.x[start:stop], dtype=np.float64)
                if not np.all(np.isfinite(block)):
                    raise ValueError("X contains NaN or infinite values.")
                total += float(np.einsum("ij,ij->", block, block))
            self.norm_sq = total
        self.norm_sq = max(self.norm_sq, 0.0)

    def _batches(self):
        for start in range(0, self.n, self.batch_size):
            yield start, min(start + self.batch_size, self.n)

    def right(self, b: ArrayLike) -> FloatArray:
        """Return centered ``X @ B`` without materializing centered X."""

        b_arr = np.asarray(b, dtype=np.float64)
        was_vector = b_arr.ndim == 1
        if was_vector:
            b_arr = b_arr[:, None]
        if b_arr.ndim != 2 or b_arr.shape[0] != self.p:
            raise ValueError(f"Right multiplier must have shape ({self.p}, q).")
        result = np.empty((self.n, b_arr.shape[1]), dtype=np.float64)
        correction = self.mean @ b_arr
        for start, stop in self._batches():
            result[start:stop] = np.asarray(self.x[start:stop]) @ b_arr - correction
        return result[:, 0] if was_vector else result

    def transpose_right(self, a: ArrayLike) -> FloatArray:
        """Return centered ``X.T @ A`` without materializing centered X."""

        a_arr = np.asarray(a, dtype=np.float64)
        was_vector = a_arr.ndim == 1
        if was_vector:
            a_arr = a_arr[:, None]
        if a_arr.ndim != 2 or a_arr.shape[0] != self.n:
            raise ValueError(f"Multiplier must have shape ({self.n}, q).")
        result = np.zeros((self.p, a_arr.shape[1]), dtype=np.float64)
        for start, stop in self._batches():
            result += np.asarray(self.x[start:stop]).T @ a_arr[start:stop]
        result -= self.mean[:, None] * np.sum(a_arr, axis=0)[None, :]
        return result[:, 0] if was_vector else result


@dataclass
class InnerSolverInfo:
    iterations: int
    relative_change: float
    primal_dual_residual: float
    initial_objective: float
    final_objective: float
    accepted: bool
    converged: bool
    tolerance: float
    tau: float
    sigma: float


@dataclass
class OuterIterationInfo:
    iteration: int
    objective: float
    reconstruction_error: float
    explained_variance: float
    relative_objective_change: float
    relative_reconstruction_change: float
    lipschitz_u: float
    condition_v: float
    k_eff: int
    reinitialized: tuple[int, ...]
    inner: InnerSolverInfo


class CalendarGraphFusedSparsePCA:
    """Joint non-orthogonal sparse PCA with toroidal graph total variation.

    Parameters
    ----------
    n_components:
        Nominal rank ``K``.
    lambda_l1, lambda_tv:
        Loading sparsity and graph-fusion penalties.
    calendar_shape:
        Calendar tensor dimensions. The paper model uses ``(24, 7, 52)``.
    order:
        ``"C"`` means week varies fastest (NumPy convention); ``"F"`` is
        compatible with vectors flattened from an R/Fortran array.
    score_solver:
        ``"coordinate"`` is the mathematically consistent default. It solves
        every unit-sphere score-coordinate subproblem exactly and never rescales
        ``V``. ``"rescaled_ls"`` reproduces the pseudoinverse/rescaling update
        in the supplied formulation, but that update can increase the penalized
        objective and is retained only for experimentation.
    """

    def __init__(
        self,
        n_components: int,
        *,
        lambda_l1: float = 0.0,
        lambda_tv: float = 0.0,
        calendar_shape: tuple[int, int, int] = DEFAULT_CALENDAR_SHAPE,
        order: Literal["C", "F"] = "C",
        center: bool = True,
        score_solver: Literal["coordinate", "rescaled_ls"] = "coordinate",
        score_sweeps: int = 10,
        score_tolerance: float = 1e-7,
        outer_max_iter: int = 100,
        outer_objective_tolerance: float = 1e-6,
        outer_reconstruction_tolerance: float = 1e-6,
        inner_max_iter: int = 20_000,
        inner_check_interval: int = 25,
        epsilon_0: float = 1e-3,
        epsilon_min: float = 1e-6,
        epsilon_rho: float = 0.75,
        step_safety: float = 0.99,
        dual_step_scale: float = 1.0,
        loading_collapse_tolerance: float = 1e-10,
        score_collapse_tolerance: float = 1e-12,
        similarity_threshold: float = 0.995,
        condition_threshold: float = 1e12,
        max_reinitializations: int = 3,
        reinit_power_iterations: int = 5,
        initialization: Literal["svd", "random"] = "svd",
        svd_power_iterations: int = 2,
        svd_oversamples: int = 5,
        batch_size: int = 2048,
        sparsity_tolerance: float = 1e-10,
        effective_region_min_active_fraction: float = 0.005,
        effective_region_min_l1_fraction: float = 0.01,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_tv = float(lambda_tv)
        self.calendar_shape = tuple(int(s) for s in calendar_shape)
        self.order = order
        self.center = bool(center)
        self.score_solver = score_solver
        self.score_sweeps = int(score_sweeps)
        self.score_tolerance = float(score_tolerance)
        self.outer_max_iter = int(outer_max_iter)
        self.outer_objective_tolerance = float(outer_objective_tolerance)
        self.outer_reconstruction_tolerance = float(outer_reconstruction_tolerance)
        self.inner_max_iter = int(inner_max_iter)
        self.inner_check_interval = int(inner_check_interval)
        self.epsilon_0 = float(epsilon_0)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_rho = float(epsilon_rho)
        self.step_safety = float(step_safety)
        self.dual_step_scale = float(dual_step_scale)
        self.loading_collapse_tolerance = float(loading_collapse_tolerance)
        self.score_collapse_tolerance = float(score_collapse_tolerance)
        self.similarity_threshold = float(similarity_threshold)
        self.condition_threshold = float(condition_threshold)
        self.max_reinitializations = int(max_reinitializations)
        self.reinit_power_iterations = int(reinit_power_iterations)
        self.initialization = initialization
        self.svd_power_iterations = int(svd_power_iterations)
        self.svd_oversamples = int(svd_oversamples)
        self.batch_size = int(batch_size)
        self.sparsity_tolerance = float(sparsity_tolerance)
        self.effective_region_min_active_fraction = float(
            effective_region_min_active_fraction
        )
        self.effective_region_min_l1_fraction = float(
            effective_region_min_l1_fraction
        )
        self.random_state = random_state
        self.verbose = int(verbose)

    def _validate_parameters(self, p: int, n: int) -> None:
        if self.n_components < 1 or self.n_components > min(n, p):
            raise ValueError("n_components must be between 1 and min(n, p).")
        if self._xop.norm_sq <= np.finfo(float).tiny:
            raise ValueError("X has no variation after the requested centering.")
        if self.lambda_l1 < 0 or self.lambda_tv < 0:
            raise ValueError("lambda_l1 and lambda_tv must be non-negative.")
        if self.graph_.p != p:
            raise ValueError(
                f"X has {p} columns, but calendar_shape={self.calendar_shape} "
                f"contains {self.graph_.p} cells."
            )
        if self.score_solver not in {"coordinate", "rescaled_ls"}:
            raise ValueError("Unknown score_solver.")
        if self.initialization not in {"svd", "random"}:
            raise ValueError("initialization must be 'svd' or 'random'.")
        if not (0 < self.epsilon_rho < 1):
            raise ValueError("epsilon_rho must lie strictly between 0 and 1.")
        if not (0 < self.step_safety < 1):
            raise ValueError("step_safety must lie strictly between 0 and 1.")
        if self.dual_step_scale <= 0:
            raise ValueError("dual_step_scale must be positive.")
        if not (0 < self.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must lie in (0, 1].")
        if self.condition_threshold <= 1:
            raise ValueError("condition_threshold must be greater than 1.")
        fractions = {
            "effective_region_min_active_fraction": (
                self.effective_region_min_active_fraction
            ),
            "effective_region_min_l1_fraction": self.effective_region_min_l1_fraction,
        }
        for name, value in fractions.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1].")
        positive_ints = {
            "score_sweeps": self.score_sweeps,
            "outer_max_iter": self.outer_max_iter,
            "inner_max_iter": self.inner_max_iter,
            "inner_check_interval": self.inner_check_interval,
            "batch_size": self.batch_size,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be positive.")

    def _randomized_svd_initialization(self) -> tuple[FloatArray, FloatArray]:
        k = self.n_components
        ell = min(self._xop.p, k + max(0, self.svd_oversamples))
        omega = self._rng.normal(size=(self._xop.p, ell))
        q, _ = np.linalg.qr(self._xop.right(omega), mode="reduced")
        for _ in range(max(0, self.svd_power_iterations)):
            z, _ = np.linalg.qr(self._xop.transpose_right(q), mode="reduced")
            q, _ = np.linalg.qr(self._xop.right(z), mode="reduced")
        b = self._xop.transpose_right(q).T
        u_small, singular_values, vt = np.linalg.svd(b, full_matrices=False)
        u = q @ u_small[:, :k]
        v = vt[:k].T * singular_values[:k]
        return u, v

    def _random_initialization(self) -> tuple[FloatArray, FloatArray]:
        v = self._rng.normal(size=(self._xop.p, self.n_components))
        v /= np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-15)
        xv = self._xop.right(v)
        scales = np.linalg.norm(xv, axis=0)
        u = xv / np.maximum(scales, 1e-15)
        v *= scales
        return u, v

    def _reconstruction_error(self, u: FloatArray, v: FloatArray) -> float:
        ug = u.T @ u
        vg = v.T @ v
        xtu = self._xop.transpose_right(u)
        value = self._xop.norm_sq + float(np.sum(ug * vg)) - 2.0 * float(np.sum(v * xtu))
        return max(value, 0.0)

    def _objective(self, u: FloatArray, v: FloatArray) -> float:
        return (
            0.5 * self._reconstruction_error(u, v)
            + self.lambda_l1 * float(np.sum(np.abs(v)))
            + self.lambda_tv * float(np.sum(np.abs(self.graph_.diff(v))))
        )

    @staticmethod
    def _condition_number(v: FloatArray, active: NDArray[np.bool_]) -> float:
        if np.count_nonzero(active) <= 1:
            return 1.0
        gram = v[:, active].T @ v[:, active]
        singular = np.linalg.svd(gram, compute_uv=False)
        if singular[-1] <= np.finfo(float).eps * max(singular[0], 1.0):
            return float("inf")
        return float(singular[0] / singular[-1])

    def _coordinate_score_update(self, u: FloatArray, v: FloatArray) -> set[int]:
        active_indices = np.flatnonzero(self.active_)
        collapsed: set[int] = set()
        if active_indices.size == 0:
            return set(range(self.n_components))
        xv = self._xop.right(v[:, active_indices])
        local_v = v[:, active_indices]

        for _ in range(self.score_sweeps):
            old = u[:, active_indices].copy()
            for local_k, k in enumerate(active_indices):
                vk = local_v[:, local_k]
                cross = local_v.T @ vk
                raw = xv[:, local_k] - u[:, active_indices] @ cross + u[:, k] * cross[local_k]
                norm = float(np.linalg.norm(raw))
                if norm < self.score_collapse_tolerance:
                    collapsed.add(int(k))
                    continue
                u[:, k] = raw / norm
            change = _safe_relative(
                float(np.linalg.norm(u[:, active_indices] - old)),
                float(np.linalg.norm(old)),
            )
            if change < self.score_tolerance:
                break
        return collapsed

    def _rescaled_ls_score_update(self, u: FloatArray, v: FloatArray) -> set[int]:
        warnings.warn(
            "score_solver='rescaled_ls' can change the L1/TV penalties during "
            "normalization and is not a descent-guaranteed block update.",
            RuntimeWarning,
            stacklevel=2,
        )
        indices = np.flatnonzero(self.active_)
        if indices.size == 0:
            return set(range(self.n_components))
        local_v = v[:, indices]
        raw_u = self._xop.right(local_v) @ np.linalg.pinv(local_v.T @ local_v)
        collapsed: set[int] = set()
        for local_k, k in enumerate(indices):
            scale = float(np.linalg.norm(raw_u[:, local_k]))
            if scale < self.score_collapse_tolerance:
                collapsed.add(int(k))
                continue
            u[:, k] = raw_u[:, local_k] / scale
            v[:, k] *= scale
            self.dual_[:, :, k] = 0.0
        return collapsed

    def _loading_subproblem_objective(
        self,
        v: FloatArray,
        ug: FloatArray,
        xtu: FloatArray,
    ) -> float:
        smooth = 0.5 * (
            self._xop.norm_sq
            + float(np.sum(ug * (v.T @ v)))
            - 2.0 * float(np.sum(v * xtu))
        )
        return (
            smooth
            + self.lambda_l1 * float(np.sum(np.abs(v)))
            + self.lambda_tv * float(np.sum(np.abs(self.graph_.diff(v))))
        )

    def _primal_dual_residual(
        self,
        v: FloatArray,
        y: FloatArray,
        ug: FloatArray,
        xtu: FloatArray,
        tau: float,
        sigma: float,
    ) -> float:
        grad = v @ ug - xtu
        primal_map = _soft_threshold(
            v - tau * (grad + self.graph_.adjoint(y)), tau * self.lambda_l1
        )
        dual_map = np.clip(
            y + sigma * self.graph_.diff(v), -self.lambda_tv, self.lambda_tv
        )
        primal = np.linalg.norm(v - primal_map) / max(tau, 1e-15)
        dual = np.linalg.norm(y - dual_map) / max(sigma, 1e-15)
        scale = 1.0 + np.linalg.norm(v) + np.linalg.norm(y)
        return float(np.hypot(primal, dual) / scale)

    def _update_loadings(self, u: FloatArray, v: FloatArray, tolerance: float) -> InnerSolverInfo:
        ug = u.T @ u
        xtu = self._xop.transpose_right(u)
        lipschitz = float(np.linalg.eigvalsh(ug)[-1])
        sigma = self.dual_step_scale / np.sqrt(self.graph_.norm_sq)
        tau = self.step_safety / (0.5 * lipschitz + sigma * self.graph_.norm_sq)

        if not (1.0 / tau - sigma * self.graph_.norm_sq > 0.5 * lipschitz):
            raise RuntimeError("Internal step-size calculation violated convergence condition.")

        initial = self._loading_subproblem_objective(v, ug, xtu)
        old_outer_v = v.copy()
        previous_check = v.copy()
        relative_change = float("inf")
        residual = float("inf")
        accepted = False
        converged = False
        iterations = 0
        old_dual = self.dual_.copy()

        for iteration in range(1, self.inner_max_iter + 1):
            grad = v @ ug - xtu
            new_v = _soft_threshold(
                v - tau * (grad + self.graph_.adjoint(self.dual_)),
                tau * self.lambda_l1,
            )
            extrapolated = 2.0 * new_v - v
            self.dual_ = np.clip(
                self.dual_ + sigma * self.graph_.diff(extrapolated),
                -self.lambda_tv,
                self.lambda_tv,
            )
            v = new_v
            iterations = iteration

            if iteration % self.inner_check_interval == 0 or iteration == self.inner_max_iter:
                relative_change = _safe_relative(
                    float(np.linalg.norm(v - previous_check)),
                    float(np.linalg.norm(previous_check)),
                )
                residual = self._primal_dual_residual(v, self.dual_, ug, xtu, tau, sigma)
                current = self._loading_subproblem_objective(v, ug, xtu)
                descent_ok = current <= initial + 1e-12 * (1.0 + abs(initial))
                if relative_change < tolerance and residual < tolerance and descent_ok:
                    accepted = True
                    converged = True
                    break
                previous_check = v.copy()

        final = self._loading_subproblem_objective(v, ug, xtu)
        if final <= initial + 1e-12 * (1.0 + abs(initial)):
            accepted = True
        if not accepted:
            v = old_outer_v
            self.dual_ = old_dual
            final = initial

        self.components_ = v
        return InnerSolverInfo(
            iterations=iterations,
            relative_change=float(relative_change),
            primal_dual_residual=float(residual),
            initial_objective=float(initial),
            final_objective=float(final),
            accepted=accepted,
            converged=converged,
            tolerance=float(tolerance),
            tau=float(tau),
            sigma=float(sigma),
        )

    def _conditional_contributions(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Increase in SSE after deleting each fitted rank-one term."""

        xv = self._xop.right(v)
        ug = u.T @ u
        vg = v.T @ v
        residual_inner = np.diag(u.T @ xv) - np.sum(ug * vg, axis=1)
        return 2.0 * residual_inner + np.diag(ug) * np.diag(vg)

    def _project_out(self, candidate: FloatArray, v: FloatArray, indices: NDArray[np.int_]) -> FloatArray:
        if indices.size == 0:
            return candidate
        basis = v[:, indices]
        return candidate - basis @ (np.linalg.pinv(basis.T @ basis) @ (basis.T @ candidate))

    def _reinitialize_component(self, k: int, u: FloatArray, v: FloatArray) -> bool:
        if self.reinitialization_counts_[k] >= self.max_reinitializations:
            self.active_[k] = False
            u[:, k] = 0.0
            v[:, k] = 0.0
            self.dual_[:, :, k] = 0.0
            return False

        self.reinitialization_counts_[k] += 1
        other = np.flatnonzero(self.active_ & (np.arange(self.n_components) != k))
        candidate = self._rng.normal(size=self._xop.p)
        candidate = self._project_out(candidate, v, other)
        norm = np.linalg.norm(candidate)
        if norm < 1e-14:
            self.active_[k] = False
            return False
        candidate /= norm

        for _ in range(max(1, self.reinit_power_iterations)):
            left = self._xop.right(candidate)
            if other.size:
                left -= u[:, other] @ (v[:, other].T @ candidate)
            right = self._xop.transpose_right(left)
            if other.size:
                right -= v[:, other] @ (u[:, other].T @ left)
            candidate = self._project_out(right, v, other)
            norm = np.linalg.norm(candidate)
            if norm < 1e-14:
                self.active_[k] = False
                u[:, k] = 0.0
                v[:, k] = 0.0
                return False
            candidate /= norm

        left = self._xop.right(candidate)
        if other.size:
            left -= u[:, other] @ (v[:, other].T @ candidate)
        amplitude = float(np.linalg.norm(left))
        if amplitude < self.score_collapse_tolerance:
            self.active_[k] = False
            u[:, k] = 0.0
            v[:, k] = 0.0
            return False
        self.active_[k] = True
        u[:, k] = left / amplitude
        v[:, k] = amplitude * candidate
        self.dual_[:, :, k] = 0.0
        return True

    def _handle_degeneracy(self, u: FloatArray, v: FloatArray, raw_score_bad: set[int]) -> set[int]:
        reinitialized: set[int] = set()
        loading_norms = np.linalg.norm(v, axis=0)
        bad = set(np.flatnonzero(self.active_ & (loading_norms < self.loading_collapse_tolerance)))
        bad.update(raw_score_bad)
        for k in sorted(bad):
            if self._reinitialize_component(k, u, v):
                reinitialized.add(k)

        indices = np.flatnonzero(self.active_)
        if indices.size > 1:
            local = v[:, indices]
            norms = np.linalg.norm(local, axis=0)
            similarity = np.abs(local.T @ local) / np.maximum(norms[:, None] * norms[None, :], 1e-30)
            contributions = self._conditional_contributions(u, v)
            duplicate_pairs = np.argwhere(np.triu(similarity, 1) > self.similarity_threshold)
            already = set()
            for local_i, local_j in duplicate_pairs:
                i, j = int(indices[local_i]), int(indices[local_j])
                if i in already or j in already or not (self.active_[i] and self.active_[j]):
                    continue
                loser = i if contributions[i] < contributions[j] else j
                if self._reinitialize_component(loser, u, v):
                    reinitialized.add(loser)
                already.add(loser)

        for _ in range(self.n_components):
            indices = np.flatnonzero(self.active_)
            if indices.size <= 1:
                break
            local = v[:, indices]
            gram = local.T @ local
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            largest = max(float(eigenvalues[-1]), np.finfo(float).tiny)
            smallest = max(float(eigenvalues[0]), 0.0)
            condition = float("inf") if smallest <= np.finfo(float).eps * largest else largest / smallest
            if condition <= self.condition_threshold:
                break
            dependency = np.abs(eigenvectors[:, 0])
            implicated_local = np.flatnonzero(dependency >= 0.5 * np.max(dependency))
            if implicated_local.size == 0:
                implicated_local = np.array([int(np.argmax(dependency))])
            implicated = indices[implicated_local]
            contributions = self._conditional_contributions(u, v)
            loser = int(implicated[np.argmin(contributions[implicated])])
            if self._reinitialize_component(loser, u, v):
                reinitialized.add(loser)
            else:
                continue
        return reinitialized

    def fit(self, x: ArrayLike) -> "CalendarGraphFusedSparsePCA":
        """Fit the model to an ``(n, H*D*W)`` numeric matrix."""

        self.graph_ = ToroidalCalendarGraph(self.calendar_shape, order=self.order)
        self._xop = _CenteredMatrix(x, self.center, self.batch_size)
        self._validate_parameters(self._xop.p, self._xop.n)
        self._rng = np.random.default_rng(self.random_state)
        self.mean_ = self._xop.mean.copy()
        self.total_sum_squares_ = self._xop.norm_sq
        self.active_ = np.ones(self.n_components, dtype=bool)
        self.reinitialization_counts_ = np.zeros(self.n_components, dtype=int)

        if self.initialization == "svd":
            u, v = self._randomized_svd_initialization()
        else:
            u, v = self._random_initialization()
        self.scores_ = u
        self.components_ = v
        self.dual_ = np.zeros((3, self.graph_.p, self.n_components), dtype=np.float64)
        self.history_: list[OuterIterationInfo] = []
        self.converged_ = False

        previous_objective = self._objective(u, v)
        previous_reconstruction_norm_sq = float(np.sum((u.T @ u) * (v.T @ v)))

        for outer in range(self.outer_max_iter):
            if self.score_solver == "coordinate":
                raw_score_bad = self._coordinate_score_update(u, v)
            else:
                raw_score_bad = self._rescaled_ls_score_update(u, v)

            tolerance = max(self.epsilon_min, self.epsilon_0 * self.epsilon_rho**outer)
            inner = self._update_loadings(u, v, tolerance)
            v = self.components_
            reinitialized = self._handle_degeneracy(u, v, raw_score_bad)

            objective = self._objective(u, v)
            reconstruction_error = self._reconstruction_error(u, v)
            explained = 1.0 - reconstruction_error / max(self.total_sum_squares_, 1e-30)
            relative_objective = _safe_relative(
                abs(objective - previous_objective), abs(previous_objective)
            )

            if outer == 0:
                relative_reconstruction = float("inf")
            else:
                old_norm = previous_reconstruction_norm_sq
                new_norm = float(np.sum((u.T @ u) * (v.T @ v)))
                cross = float(np.sum((previous_u.T @ u) * (previous_v.T @ v)))
                difference = np.sqrt(max(old_norm + new_norm - 2.0 * cross, 0.0))
                relative_reconstruction = _safe_relative(difference, np.sqrt(max(old_norm, 0.0)))

            lipschitz = float(np.linalg.eigvalsh(u.T @ u)[-1])
            condition = self._condition_number(v, self.active_)
            info = OuterIterationInfo(
                iteration=outer + 1,
                objective=float(objective),
                reconstruction_error=float(reconstruction_error),
                explained_variance=float(explained),
                relative_objective_change=float(relative_objective),
                relative_reconstruction_change=float(relative_reconstruction),
                lipschitz_u=lipschitz,
                condition_v=condition,
                k_eff=int(np.count_nonzero(self.active_)),
                reinitialized=tuple(sorted(reinitialized)),
                inner=inner,
            )
            self.history_.append(info)
            if self.verbose:
                print(
                    f"outer={outer + 1:03d} J={objective:.6e} "
                    f"EV={explained:.5f} K_eff={info.k_eff} "
                    f"inner={inner.iterations} reinit={info.reinitialized}"
                )

            if (
                outer > 0
                and relative_objective < self.outer_objective_tolerance
                and relative_reconstruction < self.outer_reconstruction_tolerance
                and not reinitialized
                and inner.converged
            ):
                self.converged_ = True
                break

            previous_u = u.copy()
            previous_v = v.copy()
            previous_reconstruction_norm_sq = float(np.sum((u.T @ u) * (v.T @ v)))
            previous_objective = objective

        self.n_iter_ = len(self.history_)
        self.scores_ = u
        self.components_ = v
        self.n_components_effective_ = int(np.count_nonzero(self.active_))
        self._apply_sign_convention()
        self._finalize_diagnostics()
        return self

    def get_params(self) -> dict[str, Any]:
        """Return constructor parameters without copying fitted data."""

        names = (
            "n_components",
            "lambda_l1",
            "lambda_tv",
            "calendar_shape",
            "order",
            "center",
            "score_solver",
            "score_sweeps",
            "score_tolerance",
            "outer_max_iter",
            "outer_objective_tolerance",
            "outer_reconstruction_tolerance",
            "inner_max_iter",
            "inner_check_interval",
            "epsilon_0",
            "epsilon_min",
            "epsilon_rho",
            "step_safety",
            "dual_step_scale",
            "loading_collapse_tolerance",
            "score_collapse_tolerance",
            "similarity_threshold",
            "condition_threshold",
            "max_reinitializations",
            "reinit_power_iterations",
            "initialization",
            "svd_power_iterations",
            "svd_oversamples",
            "batch_size",
            "sparsity_tolerance",
            "effective_region_min_active_fraction",
            "effective_region_min_l1_fraction",
            "random_state",
            "verbose",
        )
        return {name: getattr(self, name) for name in names}

    def _apply_sign_convention(self) -> None:
        for k in np.flatnonzero(self.active_):
            index = int(np.argmax(np.abs(self.components_[:, k])))
            if self.components_[index, k] < 0:
                self.components_[:, k] *= -1.0
                self.scores_[:, k] *= -1.0

    def _finalize_diagnostics(self) -> None:
        v = self.components_
        u = self.scores_
        norms = np.linalg.norm(v, axis=0)
        safe = np.maximum(norms, 1e-30)
        self.loading_similarity_ = np.abs(v.T @ v) / (safe[:, None] * safe[None, :])
        self.score_correlation_ = np.zeros(
            (self.n_components, self.n_components), dtype=np.float64
        )
        active_indices = np.flatnonzero(self.active_)
        if active_indices.size == 1:
            self.score_correlation_[active_indices[0], active_indices[0]] = 1.0
        elif active_indices.size > 1:
            active_corr = np.corrcoef(u[:, active_indices], rowvar=False)
            self.score_correlation_[np.ix_(active_indices, active_indices)] = active_corr
        self.conditional_contribution_ss_ = self._conditional_contributions(u, v)
        self.conditional_contributions_ = self.conditional_contribution_ss_
        self.conditional_contribution_ratio_ = (
            self.conditional_contribution_ss_ / max(self.total_sum_squares_, 1e-30)
        )
        self.conditional_contribution_percent_ = 100.0 * self.conditional_contribution_ratio_
        self.loading_sparsity_ = np.mean(np.abs(v) <= self.sparsity_tolerance, axis=0)
        self.loading_l1_norm_ = np.sum(np.abs(v), axis=0)
        self.loading_l2_norm_ = np.linalg.norm(v, axis=0)
        self.total_variation_ = self.graph_.total_variation(v)
        self.relative_total_variation_ = np.divide(
            self.total_variation_,
            self.loading_l1_norm_,
            out=np.zeros(self.n_components, dtype=np.float64),
            where=self.loading_l1_norm_ > 0,
        )
        self.active_cells_ = np.sum(np.abs(v) > self.sparsity_tolerance, axis=0, dtype=int)
        self.region_statistics_ = self.graph_.active_region_statistics(
            v, threshold=self.sparsity_tolerance
        )
        self.connected_region_sizes_ = [
            [int(region["size"]) for region in component]
            for component in self.region_statistics_
        ]
        self.n_connected_regions_ = np.array(
            [len(sizes) for sizes in self.connected_region_sizes_], dtype=int
        )
        self.effective_region_statistics_ = []
        for k, regions in enumerate(self.region_statistics_):
            active_cells = max(int(self.active_cells_[k]), 1)
            effective = [
                region
                for region in regions
                if (
                    int(region["size"]) / active_cells
                    >= self.effective_region_min_active_fraction
                    or float(region["l1_fraction"]) >= self.effective_region_min_l1_fraction
                )
            ]
            self.effective_region_statistics_.append(effective)
        self.n_effective_regions_ = np.array(
            [len(regions) for regions in self.effective_region_statistics_], dtype=int
        )
        self.largest_region_size_ = np.array(
            [int(regions[0]["size"]) if regions else 0 for regions in self.region_statistics_],
            dtype=int,
        )
        self.largest_region_active_fraction_ = np.divide(
            self.largest_region_size_,
            self.active_cells_,
            out=np.zeros(self.n_components, dtype=np.float64),
            where=self.active_cells_ > 0,
        )
        self.largest_region_l1_fraction_ = np.array(
            [float(regions[0]["l1_fraction"]) if regions else 0.0 for regions in self.region_statistics_],
            dtype=np.float64,
        )
        active_v = v[:, self.active_]
        if active_v.shape[1]:
            self.loading_gram_singular_values_ = np.linalg.svd(
                active_v.T @ active_v, compute_uv=False
            )
        else:
            self.loading_gram_singular_values_ = np.empty(0, dtype=np.float64)
        self.condition_number_ = self._condition_number(v, self.active_)
        self.reconstruction_error_ = self._reconstruction_error(u, v)
        self.explained_variance_ = 1.0 - self.reconstruction_error_ / max(
            self.total_sum_squares_, 1e-30
        )
        self.reconstruction_fraction_ = self.explained_variance_
        self.metric_denominator_ = (
            "centered_total_sum_squares" if self.center else "uncentered_sum_squares"
        )
        if active_indices.size:
            self.mean_loading_sparsity_active_ = float(np.mean(self.loading_sparsity_[active_indices]))
            self.mean_connected_regions_active_ = float(np.mean(self.n_connected_regions_[active_indices]))
            self.mean_effective_regions_active_ = float(np.mean(self.n_effective_regions_[active_indices]))
        else:
            self.mean_loading_sparsity_active_ = float("nan")
            self.mean_connected_regions_active_ = float("nan")
            self.mean_effective_regions_active_ = float("nan")

    def transform(self, x: ArrayLike) -> FloatArray:
        """Least-squares scores for new rows using the fitted loadings."""

        self._check_is_fitted()
        arr = np.asarray(x)
        if arr.ndim != 2 or arr.shape[1] != self.graph_.p:
            raise ValueError(f"x must have shape (n_samples, {self.graph_.p}).")
        centered = arr.astype(np.float64, copy=False) - self.mean_
        result = np.zeros((arr.shape[0], self.n_components), dtype=np.float64)
        indices = np.flatnonzero(self.active_)
        if indices.size:
            local_v = self.components_[:, indices]
            result[:, indices] = centered @ local_v @ np.linalg.pinv(local_v.T @ local_v)
        return result

    def inverse_transform(self, scores: ArrayLike, *, add_mean: bool = True) -> FloatArray:
        """Reconstruct profiles from component scores."""

        self._check_is_fitted()
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim != 2 or scores_arr.shape[1] != self.n_components:
            raise ValueError(f"scores must have shape (n_samples, {self.n_components}).")
        reconstructed = scores_arr @ self.components_.T
        if add_mean:
            reconstructed += self.mean_
        return reconstructed

    def fit_transform(self, x: ArrayLike) -> FloatArray:
        """Fit and return the normalized training score matrix."""

        return self.fit(x).scores_

    def loading_tensors(self) -> FloatArray:
        """Return loadings with shape ``(hour, day, week, K)``."""

        self._check_is_fitted()
        return reshape_calendar(self.components_, self.calendar_shape, order=self.order)

    def reorder_components(
        self,
        criterion: Literal["contribution", "loading_norm"] = "contribution",
    ) -> "CalendarGraphFusedSparsePCA":
        """Order active components by a stated relevance criterion."""

        self._check_is_fitted()
        if criterion == "contribution":
            relevance = self.conditional_contributions_
        elif criterion == "loading_norm":
            relevance = np.linalg.norm(self.components_, axis=0)
        else:
            raise ValueError("criterion must be 'contribution' or 'loading_norm'.")
        order = np.lexsort((-relevance, ~self.active_))
        self.scores_ = self.scores_[:, order]
        self.components_ = self.components_[:, order]
        self.dual_ = self.dual_[:, :, order]
        self.active_ = self.active_[order]
        self.reinitialization_counts_ = self.reinitialization_counts_[order]
        self._finalize_diagnostics()
        return self

    def diagnostics(self) -> dict[str, Any]:
        """Return fitted diagnostics."""

        self._check_is_fitted()
        return {
            "nominal_rank": self.n_components,
            "effective_rank": self.n_components_effective_,
            "active": self.active_.copy(),
            "converged": self.converged_,
            "n_outer_iterations": self.n_iter_,
            "explained_variance": self.explained_variance_,
            "reconstruction_fraction": self.reconstruction_fraction_,
            "metric_denominator": self.metric_denominator_,
            "total_sum_squares": self.total_sum_squares_,
            "reconstruction_error": self.reconstruction_error_,
            "conditional_contribution_ss": self.conditional_contribution_ss_.copy(),
            "conditional_contribution_ratio": self.conditional_contribution_ratio_.copy(),
            "conditional_contribution_percent": self.conditional_contribution_percent_.copy(),
            "conditional_contribution": self.conditional_contributions_.copy(),
            "loading_sparsity": self.loading_sparsity_.copy(),
            "mean_loading_sparsity_active": self.mean_loading_sparsity_active_,
            "loading_l1_norm": self.loading_l1_norm_.copy(),
            "loading_l2_norm": self.loading_l2_norm_.copy(),
            "active_cells": self.active_cells_.copy(),
            "total_variation": self.total_variation_.copy(),
            "relative_total_variation": self.relative_total_variation_.copy(),
            "n_connected_regions": self.n_connected_regions_.copy(),
            "mean_connected_regions_active": self.mean_connected_regions_active_,
            "connected_region_sizes": list(self.connected_region_sizes_),
            "region_statistics": list(self.region_statistics_),
            "n_effective_regions": self.n_effective_regions_.copy(),
            "mean_effective_regions_active": self.mean_effective_regions_active_,
            "effective_region_statistics": list(self.effective_region_statistics_),
            "largest_region_size": self.largest_region_size_.copy(),
            "largest_region_active_fraction": self.largest_region_active_fraction_.copy(),
            "largest_region_l1_fraction": self.largest_region_l1_fraction_.copy(),
            "region_support_threshold": self.sparsity_tolerance,
            "effective_region_min_active_fraction": self.effective_region_min_active_fraction,
            "effective_region_min_l1_fraction": self.effective_region_min_l1_fraction,
            "loading_similarity": self.loading_similarity_.copy(),
            "score_correlation": self.score_correlation_.copy(),
            "loading_gram_singular_values": self.loading_gram_singular_values_.copy(),
            "condition_number_v_gram": self.condition_number_,
            "reinitialization_counts": self.reinitialization_counts_.copy(),
            "operator_norm_squared": self.graph_.norm_sq,
        }

    def audit_metrics(
        self,
        *,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        """Independently recompute reconstruction metrics from data batches."""

        self._check_is_fitted()
        if relative_tolerance < 0 or absolute_tolerance < 0:
            raise ValueError("Audit tolerances must be non-negative.")

        direct_tss = 0.0
        direct_sse = 0.0
        residual_inner = np.zeros(self.n_components, dtype=np.float64)
        score_norm_sq = np.zeros(self.n_components, dtype=np.float64)
        centered_column_sum = np.zeros(self.graph_.p, dtype=np.float64)
        loading_norm_sq = np.sum(self.components_ * self.components_, axis=0)

        for start, stop in self._xop._batches():
            centered = np.array(self._xop.x[start:stop], dtype=np.float64, copy=True)
            centered -= self.mean_
            centered_column_sum += np.sum(centered, axis=0)
            direct_tss += float(np.einsum("ij,ij->", centered, centered))
            residual = centered - self.scores_[start:stop] @ self.components_.T
            direct_sse += float(np.einsum("ij,ij->", residual, residual))
            residual_inner += np.sum(
                self.scores_[start:stop] * (residual @ self.components_), axis=0
            )
            score_norm_sq += np.sum(self.scores_[start:stop] ** 2, axis=0)

        direct_contribution_ss = 2.0 * residual_inner + score_norm_sq * loading_norm_sq
        direct_ratio = 1.0 - direct_sse / max(direct_tss, 1e-30)

        def close(left: ArrayLike, right: ArrayLike) -> bool:
            return bool(np.allclose(left, right, rtol=relative_tolerance, atol=absolute_tolerance))

        region_cell_conservation = np.array(
            [
                sum(int(region["size"]) for region in regions) == int(self.active_cells_[k])
                for k, regions in enumerate(self.region_statistics_)
            ],
            dtype=bool,
        )
        active_norms = np.sqrt(score_norm_sq[self.active_])
        score_norms_valid = close(active_norms, np.ones_like(active_norms))
        inactive_indices = np.flatnonzero(~self.active_)
        inactive_zero = bool(
            np.all(self.scores_[:, inactive_indices] == 0.0)
            and np.all(self.components_[:, inactive_indices] == 0.0)
        )
        centered_column_mean_max_abs = float(np.max(np.abs(centered_column_sum / self._xop.n)))
        checks = {
            "centering_matches_requested_mode": (
                centered_column_mean_max_abs <= absolute_tolerance
                if self.center
                else bool(np.all(self.mean_ == 0.0))
            ),
            "total_sum_squares_matches_direct": close(self.total_sum_squares_, direct_tss),
            "reconstruction_error_matches_direct": close(self.reconstruction_error_, direct_sse),
            "reconstruction_fraction_matches_direct": close(self.reconstruction_fraction_, direct_ratio),
            "conditional_contributions_match_direct": close(
                self.conditional_contribution_ss_, direct_contribution_ss
            ),
            "conditional_percent_scaling_is_correct": close(
                self.conditional_contribution_percent_,
                100.0 * self.conditional_contribution_ss_ / max(self.total_sum_squares_, 1e-30),
            ),
            "active_score_norms_are_one": score_norms_valid,
            "inactive_factors_are_zero": inactive_zero,
            "region_sizes_sum_to_active_cells": bool(np.all(region_cell_conservation)),
            "all_reported_values_are_finite": bool(
                np.isfinite([direct_tss, direct_sse, direct_ratio, self.condition_number_]).all()
                and np.isfinite(self.conditional_contribution_percent_).all()
            ),
        }
        return {
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "metric_denominator": self.metric_denominator_,
            "direct_total_sum_squares": direct_tss,
            "formula_total_sum_squares": self.total_sum_squares_,
            "direct_reconstruction_error": direct_sse,
            "formula_reconstruction_error": self.reconstruction_error_,
            "direct_reconstruction_fraction": direct_ratio,
            "formula_reconstruction_fraction": self.reconstruction_fraction_,
            "direct_conditional_contribution_ss": direct_contribution_ss,
            "formula_conditional_contribution_ss": self.conditional_contribution_ss_.copy(),
            "centered_column_mean_max_abs": centered_column_mean_max_abs,
            "region_cell_conservation_by_component": region_cell_conservation,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        }

    def history_as_dicts(self) -> list[dict[str, Any]]:
        """JSON/data-frame friendly optimization history."""

        self._check_is_fitted()
        return [asdict(item) for item in self.history_]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before using this method.")
