# Calendar-SPCA method

This document summarizes the mathematical formulation and numerical conventions implemented in the `calendar_spca` package. The accompanying paper provides the scientific motivation, experimental design, and empirical analysis.

## 1. Calendar representation

Let

$$
X \in \mathbb{R}^{N\times M}
$$

contain `N` observations and `M` ordered features. Calendar-SPCA assumes that the feature axis can be reshaped into three known periodic coordinates,

$$
M=m_1m_2m_3,
$$

with calendar tensor shape

$$
(m_1,m_2,m_3).
$$

The electricity experiments use:

- GoiEner: `(24, 7, 52)` for hour of day, weekday, and ISO week;
- Low Carbon London: `(48, 7, 52)` for half-hour slot, weekday, and ISO week.

The package supports both NumPy/C and Fortran/R flattening conventions through the `order` argument. The same convention is used by all graph operators and loading reshaping utilities.

By default, Calendar-SPCA centers each feature across observations. The centered matrix is denoted

$$
X_c = X - \mathbf{1}\mu^\top.
$$

The implementation evaluates products with `X_c` in row batches and does not require materializing a second centered copy of the complete data matrix.

## 2. Cyclic Cartesian-product graph

The feature domain is represented by

$$
G=C_{m_1}\square C_{m_2}\square C_{m_3},
$$

where every axis is cyclic. Each calendar position therefore has two neighbours along each coordinate. In the hourly representation, for example, the graph connects:

- 23:00 with 00:00;
- Sunday with Monday;
- week 52 with week 1.

Let `D_G` denote an oriented incidence operator. The implementation stores no sparse incidence matrix. Instead, forward cyclic differences are evaluated directly on reshaped loading tensors with `numpy.roll`, and the exact adjoint is implemented by the corresponding backward cyclic shifts.

For the three-axis product graph, the squared spectral norm is available analytically as

$$ \lVert D_G\rVert_2^2 = \sum_{r=1}^{3}\lambda_{\max}(L_{C_{m_r}}). $$

where

$$
\lambda_{\max}(L_{C_m})=
\begin{cases}
4, & m \text{ even},\\
2+2\cos(\pi/m), & m \text{ odd}.
\end{cases}
$$

This value is used directly in the primal-dual step-size construction.

## 3. Joint sparse factorization

Calendar-SPCA estimates a rank-`K` representation

$$
X_c \approx UV^\top,
$$

with

$$
U\in\mathbb{R}^{N\times K},
\qquad
V\in\mathbb{R}^{M\times K}.
$$

The fitted factors minimize

$$ \frac12\lVert X_c-UV^\top\rVert_F^2 + \lambda_1\lVert V\rVert_{1,1} + \lambda_{\mathrm{TV}}\lVert D_GV\rVert_{1,1}. $$

subject to unit Euclidean norm for each active score column,

$$
\lVert u_k\rVert_2=1.
$$

The two structural terms play complementary roles:

- `lambda_l1` promotes selective loading support;
- `lambda_tv` promotes locally coherent values over neighbouring calendar positions.

All `K` factors are updated within the same reconstruction objective.

## 4. Alternating optimization

The implementation alternates between score-coordinate updates and a joint regularized loading update.

### 4.1 Score update

For fixed `V`, active score columns are updated sequentially on the unit sphere. For component `k`, the current contribution of the remaining factors is removed and the residual direction associated with `v_k` is normalized. Several coordinate sweeps may be performed until the score change reaches the configured tolerance.

The fitted score columns therefore maintain unit Euclidean norm throughout the standard optimization path.

### 4.2 Loading update

For fixed `U`, the loading subproblem is convex:

$$ \min_V \frac12\lVert X_c-UV^\top\rVert_F^2 + \lambda_1\lVert V\rVert_{1,1} + \lambda_{\mathrm{TV}}\lVert D_GV\rVert_{1,1}. $$

Its smooth gradient is

$$
\nabla f(V)=V(U^\top U)-X_c^\top U.
$$

The nonsmooth terms are handled by a primal-dual three-operator update. The `L1` proximal map is soft thresholding, while the dual variable associated with graph total variation is projected onto

$$
[-\lambda_{\mathrm{TV}},\lambda_{\mathrm{TV}}].
$$

Let

$$
L=\lambda_{\max}(U^\top U).
$$

The default steps are constructed from the exact graph norm as

$$
\sigma=\frac{s_{\mathrm{dual}}}{\lVert D_G\rVert_2},
$$

and

$$
\tau=\frac{s_{\mathrm{step}}}{L/2+\sigma\lVert D_G\rVert_2^2},
$$

with a safety factor `s_step < 1`. The implementation verifies the corresponding strict convergence inequality before entering the inner iterations.

### 4.3 Progressive inner accuracy

The inner tolerance at outer iteration `t` is

$$ \varepsilon_t = \max\left(\varepsilon_{\min},\varepsilon_0\rho^t\right), \qquad 0<\rho<1. $$

This schedule gives inexpensive early loading updates and progressively tighter solutions as the alternating procedure approaches a stable factorization.

The inner solver checks both relative loading change and a normalized primal-dual residual. An update is accepted only when the loading-subproblem objective satisfies the configured descent safeguard.

## 5. Convergence and component recovery

The outer iteration monitors:

- relative change in the complete penalized objective;
- relative change in the reconstructed matrix;
- convergence of the inner loading update;
- component reinitialization events.

A fit is declared converged when both outer changes satisfy their tolerances, the current inner problem is converged, and no component has been reinitialized during the iteration.

Calendar-SPCA also tracks an effective rank `K_eff`. Components may be recovered when their loading or score contribution collapses, when two loading directions become nearly duplicate, or when the active loading Gram matrix becomes severely ill-conditioned. Recovery uses a residual-informed randomized direction followed by short power iterations and projection away from the remaining active loading span. A component that exhausts its allowed recovery attempts is marked inactive.

These safeguards keep the nominal rank and effective rank conceptually separate and make rank loss explicit in the reported diagnostics.

## 6. Initialization

The default initialization is a randomized low-rank SVD computed through matrix products with the centered data operator. The number of power iterations and oversampling directions are configurable.

A random initialization is also available. All stochastic operations use NumPy's `Generator` with the supplied `random_state`.

For controlled experiment grids, using the same initialization seed across compared regularization settings enables paired comparisons while preserving independent seeds across repetitions, ranks, and sample sizes.

## 7. Diagnostics

After fitting, the estimator reports reconstruction and structural diagnostics for every component and for the active representation.

### Reconstruction

Explained variance is

$$ \operatorname{EV} = 1-\frac{\lVert X_c-UV^\top\rVert_F^2}{\lVert X_c\rVert_F^2}. $$

Conditional component contribution is evaluated as the increase in squared reconstruction error obtained by deleting one fitted rank-one term while keeping the remaining fitted terms fixed.

### Loading sparsity

For numerical tolerance `eps`, component sparsity is the fraction of loading entries satisfying

$$
|v_{jk}|\leq \text{eps}.
$$

### Calendar coherence

For each component,

$$ \operatorname{RTV}(v_k) = \frac{\lVert D_Gv_k\rVert_1}{\lVert v_k\rVert_1}. $$

Lower values indicate smaller loading variation across neighbouring calendar positions.

### Connected regions

The active loading support is decomposed into connected components on the same cyclic calendar graph. The implementation reports region sizes, loading mass, and effective-region summaries based on configurable support and loading-mass thresholds.

### Stability

`match_components` aligns two loading systems by maximum total absolute cosine similarity using the Hungarian assignment algorithm. This accounts for component permutation and sign indeterminacy.

## 8. Matrix-free computation

The implementation is designed for high-dimensional calendar matrices and avoids several unnecessary large intermediate objects:

- centered data are accessed in row batches;
- graph differences and adjoints are computed by cyclic tensor shifts;
- reconstruction error is evaluated from Gram matrices and cross-products;
- full reconstructed `N x M` matrices are not required during optimization.

The `batch_size` parameter controls the principal data-access block size.

## 9. Main public interface

The recommended estimator name is

```python
from calendar_spca import CalendarSPCA
```

A typical fit is

```python
model = CalendarSPCA(
    n_components=15,
    lambda_l1=2.0,
    lambda_tv=2.0,
    calendar_shape=(24, 7, 52),
    order="F",
    center=True,
    random_state=42,
).fit(X)
```

The main fitted attributes include:

- `scores_`: score matrix `U`;
- `components_`: loading matrix `V`;
- `active_`: active-component mask;
- `n_components_effective_`: effective rank;
- `explained_variance_`;
- `loading_sparsity_`;
- `relative_total_variation_`;
- `region_statistics_`;
- `history_`: outer-iteration diagnostics.

The `diagnostics()` and `audit_metrics()` methods provide structured summaries and independent reconstruction checks.

## 10. Reproducibility

A reproducible Calendar-SPCA fit should record:

- software version;
- calendar shape and flattening order;
- preprocessing and centering convention;
- nominal rank;
- `lambda_l1` and `lambda_tv`;
- initialization seed;
- outer and inner iteration limits;
- convergence tolerances;
- input-data provenance and, where possible, an immutable file hash.

The experiment directories in this repository contain the configurations used for the principal analyses reported in the paper.
