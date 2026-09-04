from __future__ import annotations

import unittest

import numpy as np

from calendar_spca import (
    CalendarSPCA,
    ToroidalCalendarGraph,
    flatten_calendar,
    match_components,
    reshape_calendar,
)


class GraphTests(unittest.TestCase):
    def test_flatten_round_trip_c_and_f(self):
        rng = np.random.default_rng(1)
        tensor = rng.normal(size=(4, 3, 5, 2))
        for order in ("C", "F"):
            flat = flatten_calendar(tensor, order=order)
            recovered = reshape_calendar(flat, (4, 3, 5), order=order)
            np.testing.assert_allclose(recovered, tensor)

    def test_adjoint_identity(self):
        rng = np.random.default_rng(2)
        graph = ToroidalCalendarGraph((4, 3, 6))
        x = rng.normal(size=(graph.p, 3))
        y = rng.normal(size=(3, graph.p, 3))
        lhs = np.sum(graph.diff(x) * y)
        rhs = np.sum(x * graph.adjoint(y))
        self.assertAlmostEqual(lhs, rhs, places=11)

    def test_exact_norm_for_paper_calendar(self):
        graph = ToroidalCalendarGraph()
        expected = 10.0 + 2.0 * np.cos(np.pi / 7.0)
        self.assertAlmostEqual(graph.norm_sq, expected, places=13)

    def test_regions_wrap_across_boundary(self):
        graph = ToroidalCalendarGraph((4, 3, 5))
        tensor = np.zeros((4, 3, 5))
        tensor[0, 1, 0] = 1.0
        tensor[-1, 1, 0] = 1.0
        tensor[0, 1, -1] = 1.0
        sizes = graph.connected_active_regions(flatten_calendar(tensor))
        self.assertEqual(sizes, [[3]])

    def test_region_statistics_account_for_size_and_mass(self):
        graph = ToroidalCalendarGraph((5, 5, 5))
        tensor = np.zeros((5, 5, 5))
        tensor[0:2, 0:2, 0:2] = 2.0
        tensor[3, 3, 3] = 0.25
        statistics = graph.active_region_statistics(flatten_calendar(tensor))
        self.assertEqual([region["size"] for region in statistics[0]], [8, 1])
        self.assertAlmostEqual(
            sum(float(region["l1_fraction"]) for region in statistics[0]),
            1.0,
        )
        self.assertAlmostEqual(
            sum(float(region["l2_energy_fraction"]) for region in statistics[0]),
            1.0,
        )


class EstimatorTests(unittest.TestCase):
    @staticmethod
    def synthetic():
        rng = np.random.default_rng(4)
        shape = (5, 3, 4)
        n = 80
        a = np.zeros(shape)
        a[1:4, 0:2, 1:3] = 1.0
        b = np.zeros(shape)
        b[[4, 0], 2, [3, 0]] = 1.0
        v = np.column_stack([flatten_calendar(a), flatten_calendar(b)])
        u = rng.normal(size=(n, 2))
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        x = u @ (8.0 * v).T + 0.08 * rng.normal(size=(n, v.shape[0]))
        x += rng.normal(scale=0.3, size=v.shape[0])
        return x, shape

    def test_fit_and_diagnostics(self):
        x, shape = self.synthetic()
        model = CalendarSPCA(
            2,
            lambda_l1=0.01,
            lambda_tv=0.02,
            calendar_shape=shape,
            outer_max_iter=20,
            inner_max_iter=3000,
            inner_check_interval=10,
            epsilon_0=1e-4,
            epsilon_min=1e-6,
            random_state=5,
        ).fit(x)

        self.assertEqual(model.scores_.shape, (x.shape[0], 2))
        self.assertEqual(model.components_.shape, (x.shape[1], 2))
        np.testing.assert_allclose(
            np.linalg.norm(model.scores_[:, model.active_], axis=0), 1.0, atol=1e-9
        )
        self.assertGreater(model.explained_variance_, 0.9)
        self.assertTrue(np.isfinite(model.history_[-1].objective))
        inner = model.history_[-1].inner
        self.assertGreater(
            1.0 / inner.tau - inner.sigma * model.graph_.norm_sq,
            model.history_[-1].lipschitz_u / 2.0,
        )
        expected_norm_sq = (
            (2.0 + 2.0 * np.cos(np.pi / 5.0))
            + (2.0 + 2.0 * np.cos(np.pi / 3.0))
            + 4.0
        )
        self.assertAlmostEqual(model.graph_.norm_sq, expected_norm_sq, places=12)
        self.assertEqual(model.loading_tensors().shape, (*shape, 2))
        diagnostics = model.diagnostics()
        self.assertIn("conditional_contribution", diagnostics)
        self.assertIn("conditional_contribution_percent", diagnostics)
        np.testing.assert_allclose(
            diagnostics["conditional_contribution_percent"],
            100.0
            * diagnostics["conditional_contribution_ss"]
            / diagnostics["total_sum_squares"],
        )
        self.assertIn("n_effective_regions", diagnostics)
        for active_cells, sizes in zip(
            diagnostics["active_cells"], diagnostics["connected_region_sizes"]
        ):
            self.assertEqual(int(active_cells), sum(sizes))
        self.assertEqual(
            diagnostics["loading_gram_singular_values"].size,
            model.n_components_effective_,
        )
        audit = model.audit_metrics(relative_tolerance=1e-8, absolute_tolerance=1e-7)
        self.assertTrue(audit["all_checks_passed"], audit["checks"])

    def test_training_reconstruction_formula(self):
        x, shape = self.synthetic()
        model = CalendarSPCA(
            2,
            calendar_shape=shape,
            outer_max_iter=2,
            inner_max_iter=100,
            inner_check_interval=5,
            random_state=6,
        ).fit(x)
        centered = x - x.mean(axis=0)
        direct = np.linalg.norm(centered - model.scores_ @ model.components_.T) ** 2
        self.assertAlmostEqual(direct, model.reconstruction_error_, places=8)

    def test_conditional_contribution_matches_explicit_component_deletion(self):
        x, shape = self.synthetic()
        model = CalendarSPCA(
            2,
            calendar_shape=shape,
            outer_max_iter=3,
            inner_max_iter=100,
            inner_check_interval=5,
            random_state=8,
        ).fit(x)
        centered = x - x.mean(axis=0)
        full_reconstruction = model.scores_ @ model.components_.T
        full_sse = np.linalg.norm(centered - full_reconstruction) ** 2
        explicit = np.empty(model.n_components)
        for k in range(model.n_components):
            without_k = full_reconstruction - np.outer(
                model.scores_[:, k], model.components_[:, k]
            )
            explicit[k] = np.linalg.norm(centered - without_k) ** 2 - full_sse
        np.testing.assert_allclose(
            model.conditional_contribution_ss_, explicit, rtol=1e-10, atol=1e-8
        )

    def test_component_matching_handles_sign_and_permutation(self):
        rng = np.random.default_rng(9)
        a = rng.normal(size=(30, 3))
        b = a[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
        match = match_components(a, b)
        np.testing.assert_allclose(match.absolute_cosines, 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
