# Paper experiments

This directory contains the principal experiments supporting the Calendar-SPCA paper. It is intentionally selective: auxiliary plotting scripts, exploratory analyses, pilot runs, and development-only diagnostics are kept outside the public research release.

The three experiment families are:

- [`factorial/`](factorial/): the 5,880-fit GoiEner study over regularization, nominal rank, sample size, and repeated samples;
- [`model_selection/`](model_selection/): the rank-15 diagonal L-curve used to select the final Calendar-SPCA regularization on GoiEner and Low Carbon London;
- [`baselines/`](baselines/): the rank-matched PCA, SPCA, and SPCA-TV reference procedures used in the final cross-method comparison.

## Data locations

Machine-specific paths are never stored in the scientific configuration files. Every experiment accepts data locations through command-line arguments. For repeated use, a local `paths.json` may be created from [`paths.example.json`](paths.example.json); `paths.json` is ignored by Git.

Generated outputs are also excluded from version control by default.

## Installation

Install the repository before running an experiment:

```bash
python -m pip install -e .
```

Run each command from the repository root so that paths shown in the experiment READMEs remain valid.
