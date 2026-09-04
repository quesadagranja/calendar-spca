"""Self-contained Calendar-SPCA example on a small synthetic calendar."""

from __future__ import annotations

import numpy as np

from calendar_spca import CalendarSPCA, flatten_calendar


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    shape = (8, 4, 6)
    p = int(np.prod(shape))
    n = 160

    pattern_1 = np.zeros(shape)
    pattern_1[1:4, 0:2, 1:4] = 1.0

    pattern_2 = np.zeros(shape)
    # This region crosses both cyclic hour and week boundaries.
    pattern_2[[7, 0], 2:4, :] = 0.8
    pattern_2[[7, 0], 2:4, 1:5] = 0.0

    loadings = np.column_stack(
        [flatten_calendar(pattern_1), flatten_calendar(pattern_2)]
    )
    scores = rng.normal(size=(n, 2))
    scores /= np.linalg.norm(scores, axis=0, keepdims=True)

    x = scores @ (12.0 * loadings).T + 0.20 * rng.normal(size=(n, p))
    x += rng.uniform(0.2, 0.8, size=p)  # removed again by model centering
    return x, shape


if __name__ == "__main__":
    X, calendar_shape = make_data()

    model = CalendarSPCA(
        n_components=2,
        lambda_l1=0.10,
        lambda_tv=0.20,
        calendar_shape=calendar_shape,
        outer_max_iter=30,
        inner_max_iter=5_000,
        epsilon_0=2e-4,
        epsilon_min=1e-6,
        random_state=11,
        verbose=1,
    ).fit(X)

    print("\nFinal diagnostics")
    print("K, K_eff:", model.n_components, model.n_components_effective_)
    print("Explained variance:", round(model.explained_variance_, 5))
    print("Loading sparsity:", np.round(model.loading_sparsity_, 4))
    print("Relative TV:", np.round(model.relative_total_variation_, 4))
    print("Connected region sizes:", model.connected_region_sizes_)
