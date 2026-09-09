# Calendar-SPCA

[![DOI](https://zenodo.org/badge/1356849513.svg)](https://zenodo.org/badge/latestdoi/1356849513)

**Calendar-SPCA** is an interpretable sparse representation-learning method for data observed on multiple known periodic axes. It combines sparse principal component analysis with graph total variation on a Cartesian product of cyclic graphs, producing latent factors that are both selective and locally coherent over the calendar domain.

The method was developed for electricity-consumption profiles with daily, weekly, and annual periodic structure and is described in the manuscript:

> **Calendar-SPCA: Interpretable Representation Learning for Multi-Periodic Electricity Consumption Profiles**  
> Carlos Quesada-Granja, Tony Castillo-Calzadilla, Carlos Rizo-Maestre.  
> arXiv:2609.06060, 2026.  
> https://arxiv.org/abs/2609.06060

## Method at a glance

For a centered data matrix $X_c \in \mathbb{R}^{N\times M}$, Calendar-SPCA estimates

$$ X_c \approx U V^\top $$

through

$$ \frac{1}{2}\lVert X_c-U V^\top\rVert_F^2 + \lambda_1\lVert V\rVert_{1,1} + \lambda_{\mathrm{TV}}\lVert D_GV\rVert_{1,1}, $$

where $D_G$ is the incidence operator of a cyclic Cartesian-product graph. For the hourly electricity representation used in the paper,

$$ G=C_{24}\square C_7\square C_{52}, $$

so 23:00 and 00:00, Sunday and Monday, and week 52 and week 1 are neighbours. The Low Carbon London experiments use the analogous $C_{48}\square C_7\square C_{52}$ graph.

The loading matrix is estimated jointly across the $K$ latent factors. The implementation uses matrix-free calendar operators and a primal-dual loading update, with explicit convergence diagnostics and component-recovery safeguards.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/quesadagranja/calendar-spca.git
cd calendar-spca
python -m pip install .
```

For development:

```bash
python -m pip install -e .[dev]
```

The core package depends only on NumPy and SciPy.

## Quick start

```python
import numpy as np
from calendar_spca import CalendarSPCA

X = np.load("prepared_matrix.npy")

model = CalendarSPCA(
    n_components=15,
    lambda_l1=2.0,
    lambda_tv=2.0,
    calendar_shape=(24, 7, 52),
    order="F",
    random_state=42,
)

scores = model.fit_transform(X)
loadings = model.components_
```

Rows of `X` are observations and columns are calendar positions. The number and ordering of columns must agree with `calendar_shape` and `order`.

A small self-contained example is available in [`examples/synthetic_demo.py`](examples/synthetic_demo.py):

```bash
python examples/synthetic_demo.py
```

## Public API

The estimator is:

```python
from calendar_spca import CalendarSPCA
```

Utility functions for calendar reshaping, graph operations, component matching, and repeated-fit stability are available from the same package.

## Paper experiments

This repository includes only the experiments that support the central empirical claims of the paper. The development history and auxiliary plotting scripts are intentionally excluded.

```text
experiments/
├── factorial/          # 5,880-fit regularization/rank/sample-size study
├── model_selection/    # rank-15 diagonal L-curve selection
└── baselines/          # PCA, SPCA, and SPCA-TV comparison
```

Each experiment has its own README and accepts dataset locations as command-line arguments or through a local `paths.json` file. Machine-specific paths are not stored in the repository.

## Data

The repository does not redistribute the empirical electricity datasets. Experiment documentation describes the expected prepared inputs and the calendar representation used in the paper.

The two evaluation domains are:

- **GoiEner:** hourly annual profiles on a `24 × 7 × 52` calendar;
- **Low Carbon London:** half-hourly annual profiles on a `48 × 7 × 52` calendar.

Data preprocessing, annual-profile construction, missing-value handling, and normalization should follow the procedures described in the paper before running the corresponding experiments.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

The tests cover cyclic graph construction, flattening conventions, adjoint consistency, operator norms, connected calendar regions, estimator fitting, reconstruction accounting, diagnostics, and component matching. GitHub Actions runs the suite automatically on supported Python versions.

## Method documentation

See [`METHOD.md`](METHOD.md) for the mathematical formulation, optimization scheme, graph conventions, numerical safeguards, and reported diagnostics.

## Reproducibility conventions

Scientific parameters are version-controlled in experiment configuration files. Local data locations and generated outputs are supplied externally and are ignored by Git.

For a reported fit, record at least:

- Calendar-SPCA version;
- calendar shape and flattening order;
- number of components;
- `lambda_l1` and `lambda_tv`;
- convergence settings;
- random seed;
- input-data provenance and, where possible, an immutable file hash.

Component order and sign are not identifiable. Comparisons between fitted representations should therefore align components and account for sign changes.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The repository metadata will be updated with the paper and archived-release identifiers when they become available.

## License

Calendar-SPCA is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
