# Calendar-SPCA model selection

This directory reproduces the objective rank-15 regularization selection used for the final Calendar-SPCA models in the paper.

## Regularization path

The search is restricted to the diagonal

\[
\lambda_1=\lambda_{\mathrm{TV}}=\lambda
\]

with the pre-specified 19-point path

```text
0, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.00, 1.25,
1.50, 1.75, 2.00, 2.25, 2.50, 3, 4, 5, 7, 10
```

Every point is fitted independently with the same rank and initialization seed. Warm starts are disabled.

Dataset-specific scientific settings are stored in:

- [`config/goiener.json`](config/goiener.json)
- [`config/lcl.json`](config/lcl.json)

## L-curve quantities

For each fitted model,

\[
\rho_\lambda=\lVert X_c-UV^\top\rVert_F
\]

and

\[
\Omega_\lambda
=\lVert V\rVert_{1,1}+\lVert D_GV\rVert_{1,1}.
\]

The discrete L-curve is formed by

\[
(\log_{10}\Omega_\lambda,\log_{10}\rho_\lambda).
\]

Selection is restricted to completed, converged, full-rank (`K_eff=15`) fits with finite positive `rho` and `omega`. Among eligible interior points, `select_lambda.py` chooses the maximum three-point Menger curvature. Endpoints are not eligible for selection, and an exact curvature tie is resolved in favour of the smaller lambda.

Component visualizations and post-fit structural metrics do not enter the selection rule.

The procedure selects:

```text
GoiEner:           lambda* = 2.0
Low Carbon London: lambda* = 1.5
```

## Expected inputs

### GoiEner

The runner expects the prepared matrix used by the paper, including the three leading metadata columns. The configuration applies the `imputed <= 72` eligibility rule and exposes the following 8,736 calendar features.

### Low Carbon London

The runner expects the final normalized feature matrix with shape

```text
5359 × 17472
```

corresponding to the `(48, 7, 52)` half-hour calendar representation.

The repository does not redistribute either dataset.

## Run the path

GoiEner:

```bash
python experiments/model_selection/run_lcurve.py \
  --dataset goiener \
  --data /path/to/goiener_prepared_matrix.npy \
  --output-dir results/model_selection/goiener
```

Low Carbon London:

```bash
python experiments/model_selection/run_lcurve.py \
  --dataset lcl \
  --data /path/to/lcl_prepared_matrix.npy \
  --output-dir results/model_selection/lcl
```

The fits can also be parallelized with `--workers N`, subject to available memory and I/O bandwidth. Existing point-level JSON files are skipped, so the path is resumable.

A quick preflight is available through `--dry-run`, and `--max-points N` can be used for smoke tests.

## Select lambda

After all path points have finished:

```bash
python experiments/model_selection/select_lambda.py \
  --dataset goiener \
  --output-dir results/model_selection/goiener
```

and analogously for `lcl`.

The selection step writes:

- `lcurve.csv`: fitted path values, eligibility, curvature, and selected point;
- `selection.json`: the selected lambda and its curvature neighbourhood.
