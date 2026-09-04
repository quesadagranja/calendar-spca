# Reference methods

This directory documents the rank-15 reference representations used in the final Calendar-SPCA comparison: PCA, sparse PCA (SPCA), and sparse PCA with total variation (SPCA-TV).

The purpose of this directory is to preserve the **final comparison protocol**. Exploratory implementations, pilot runs, calibration probes, monitoring utilities, and plotting scripts used during development are intentionally excluded.

Dataset-specific selected settings are stored in:

- [`config/goiener.json`](config/goiener.json)
- [`config/lcl.json`](config/lcl.json)

The numerical values reported in the paper are collected in [`final_results.csv`](final_results.csv).

## Common evaluation

All methods are evaluated at rank 15 on the complete prepared datasets. For methods with non-orthogonal loadings, reconstruction quality is computed by least-squares projection onto the fitted loading span.

The common structural metrics are:

- loading sparsity: fraction of loading coefficients below the numerical zero tolerance;
- relative total variation (RTV): mean per-component anisotropic calendar graph TV divided by loading L1 norm.

The same cyclic calendar topology is used to evaluate all loading systems.

[`evaluate_loadings.py`](evaluate_loadings.py) applies this common evaluation to an external loading matrix.

## PCA

PCA is the rank-matched reconstruction reference and has no regularization parameter. The paper fit uses scikit-learn randomized PCA with `K=15` and `random_state=111` on the complete prepared matrix.

Install scikit-learn in the main Calendar-SPCA environment and run, for example:

```bash
python -m pip install scikit-learn
python experiments/baselines/run_pca.py \
  --dataset goiener \
  --data /path/to/goiener_prepared_matrix.npy \
  --output-dir results/baselines/goiener/pca
```

The runner stores the rank-15 loading matrix and reports explained variance together with the common calendar RTV. Use `--dataset lcl` for Low Carbon London.

## SPCA

The SPCA reference follows the alternating sparse PCA formulation of Zou et al. in its high-dimensional `arrayspc` form. The public reproduction path contains two steps.

### 1. Prepare the input matrix

The R implementation reads a portable little-endian row-major float64 binary matrix. Create it from the prepared paper data with:

```bash
python experiments/baselines/prepare_spca_input.py \
  --dataset goiener \
  --data /path/to/goiener_prepared_matrix.npy \
  --output results/baselines/goiener/spca_input.bin
```

For LCL, replace `goiener` with `lcl`.

The utility also supports deterministic subsampling through `--sample-size` and `--seed`. This is useful for reproducing the GoiEner regularization-selection sample.

### 2. Fit SPCA in R

The only additional R dependency is `irlba`.

For the complete GoiEner fit:

```bash
Rscript experiments/baselines/run_spca.R \
  --input results/baselines/goiener/spca_input.bin \
  --n 87187 \
  --p 8736 \
  --gamma 0.02 \
  --output-dir results/baselines/goiener/spca
```

For LCL, the selected scale is `gamma=0.05`, with `n=5359` and `p=17472`.

The regularization scale is expressed as

```text
threshold = gamma × tau_ref
```

where `tau_ref` is the maximum PCA zero-solution threshold computed from the rank-15 PCA initialization.

The selected values were obtained from the baseline regularization paths described in the paper. GoiEner selection used a fixed `N=5000` sample; LCL used its complete `N=5359` dataset. The selected dimensionless scale was then used for the corresponding complete-data fit.

The R runner writes the fitted loading matrix as `loadings_colmajor_f64.bin`. Evaluate it with the common metrics using:

```bash
python experiments/baselines/evaluate_loadings.py \
  --dataset goiener \
  --data /path/to/goiener_prepared_matrix.npy \
  --loadings results/baselines/goiener/spca/loadings_colmajor_f64.bin
```

## SPCA-TV

SPCA-TV uses `PCAL1L2TV` from the original ParsimonY implementation. The paper environment used:

```text
Python       3.9
NumPy        1.23.5
SciPy        1.10.1
scikit-learn 1.2.2
ParsimonY    commit 15eb99bc19a51a3e350901fdb1c90e7e68591640
```

The numerical Python dependencies are recorded in [`requirements_spca_tv.txt`](requirements_spca_tv.txt). The ParsimonY repository must be checked out at the documented commit; `run_spca_tv.py` verifies the revision by default.

SPCA-TV receives the same cyclic `C_h x C_d x C_w` neighbourhood as Calendar-SPCA through [`spca_tv_operator.py`](spca_tv_operator.py). ParsimonY retains its native isotropic TV penalty internally. Evaluation RTV is subsequently computed with the common anisotropic graph-TV definition used by the paper.

The regularization proportions follow the ParsimonY reference parameterization

```text
(l1, l2, ltv) / l1_max = alpha × (0.05, 1.00, 0.10).
```

The selected global scales are:

```text
GoiEner: alpha* = 1.5
LCL:     alpha* = 0.5
```

Run a selected fit with:

```bash
python experiments/baselines/run_spca_tv.py \
  --dataset goiener \
  --data /path/to/goiener_prepared_matrix.npy \
  --output-dir results/baselines/goiener/spca_tv
```

The runner stores the native `U`, `V`, `d` quantities and the effective loading matrix `W = V diag(d)`, and reports the common explained variance, sparsity, and RTV.

## Final comparison

`final_results.csv` is a compact, version-controlled numerical record of the cross-method values used in the manuscript. Generated model arrays and local experiment outputs remain outside Git.
