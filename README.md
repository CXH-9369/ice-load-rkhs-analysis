# Ice-Load RKHS Analysis

An open-source, specimen-balanced toolkit for nonlinear analysis of ice-load experiments using reproducing kernel Hilbert space (RKHS) methods.

## Scope

The repository contains method code only. It does **not** contain raw experiments, processed specimen data, vendor software, calibration files, paper results, personal paths, or specimen-specific configurations.

Implemented analysis modules include:

- robust pressure-state standardization and Gaussian RBF kernels;
- normalized HSIC with circular-shift inference;
- local MMD transition detection;
- exact specimen-level MMD permutation tests;
- baseline-adjusted trajectory sensitivity analysis;
- rate-blind early-contact regime discovery with RKHS k-medoids;
- kernel PCA visualization;
- event-stage pressure-field distances, bootstrap summaries and Cliff's delta;
- contact-regime-conditioned association analysis;
- composite physics-informed kernels and kernel logistic regression infrastructure;
- leave-one-specimen-out evaluation utilities.

The current validated use is nonlinear contact-state characterization and mechanism analysis. The predictive modules are research infrastructure and must not be presented as a validated fracture or peak-load predictor without leakage-free specimen-level validation.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Main entry points

```text
scripts/run_rkhs_evolution_analysis.py
scripts/run_incremental_rkhs_sensitivity.py
scripts/run_early_contact_regime_analysis.py
scripts/run_contact_regime_conditioned_analysis.py
scripts/run_event_stage_bootstrap_analysis.py
scripts/run_academic_ice_load_analysis.py
```

Run any entry point with `--help` to inspect its file contract and options. The analysis scripts expect preprocessed CSV/NPZ tables; data ingestion and preprocessing are intentionally maintained as a separate package.

## Reproducibility principles

- The independent statistical unit is the specimen, never an individual frame.
- Load states follow the first chronological crossing on the pre-peak loading branch.
- Scaling and kernel parameters must be fitted inside the training fold for predictive validation.
- Exact permutation tests preserve group sample counts.
- IQR is a descriptive interval, not a confidence interval.
- Pressure-topology transitions are data-driven state changes, not automatically crack-initiation events.

The mathematical definitions are documented in the module docstrings and implemented directly in the tested analysis functions.

## License

MIT License. See [`LICENSE`](LICENSE).
