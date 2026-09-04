# Factorial experiment

This directory reproduces the main Calendar-SPCA regularization/rank/sample-size experiment reported for GoiEner.

## Design

The fixed experiment grid is stored in [`config.json`](config.json):

- `lambda_l1`, `lambda_tv`: `{0, 0.05, 0.2, 0.5, 1, 3, 10}`;
- nominal rank `K`: `{5, 10, 15, 20, 25, 30}`;
- sample size `N`: `{5000, 10000, 15000, 20000}`;
- five repetitions.

This gives

```text
7 × 7 × 6 × 4 × 5 = 5,880 fits.
```

Each repetition uses an independent random permutation of all eligible GoiEner annual profiles. The first 20,000 observations form the master sample for that repetition, and the smaller sample sizes are nested prefixes of the same permutation.

Both the sampled observations and the initialization vary across repetitions. For a fixed repetition, sample size, and rank, all 49 regularization pairs share the same randomized-SVD initialization seed. This produces paired comparisons across the regularization grid.

## Expected input

`run.py` expects the prepared GoiEner NumPy matrix used by the paper. Its layout is recorded in `config.json`:

- columns 0--2: profile metadata;
- column 2: imputed-hour count;
- columns 3 onward: 8,736 calendar features;
- eligible observations satisfy `imputed <= 72`.

The feature domain is `(24, 7, 52)` with Fortran flattening order.

The data file is not distributed in this repository.

## Run

From the repository root:

```bash
python experiments/factorial/run.py \
  --data /path/to/goiener_prepared_matrix.npy \
  --output-dir results/factorial \
  --workers 4
```

The default is a single worker. Choose parallelism according to available memory and I/O bandwidth.

A local `experiments/paths.json` may be used instead of passing `--data` repeatedly.

### Smoke test

Inspect the design without fitting:

```bash
python experiments/factorial/run.py --data /path/to/data.npy --dry-run
```

Run only a few tasks:

```bash
python experiments/factorial/run.py \
  --data /path/to/data.npy \
  --max-tasks 3
```

### Component stability

The compact fit summaries do not store loading matrices. Add

```text
--save-components
```

when the repeat-stability analysis is required. The resulting compressed loading files can be large for the complete 5,880-fit grid and are therefore excluded from Git.

## Resume behavior

Every fit is written to its own deterministic JSON file under `OUTPUT/fits/`. Existing fit files are skipped, so interrupted runs may be restarted with the same command.

## Summarize

After the fits finish:

```bash
python experiments/factorial/summarize.py \
  --output-dir results/factorial
```

This produces:

- `fits.csv`: one row per completed fit;
- `factorial_summary.csv`: one row per `(N, K, lambda_l1, lambda_tv)` configuration.

The summary includes the number of converged fits, the paper reliability indicator (`>=4` converged fits among the five repetitions), mean reconstruction/structure metrics, and pairwise component similarity when saved loading matrices are available.
