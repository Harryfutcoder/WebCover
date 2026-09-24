# Observation Metrics (Paper-Aligned)

This project logs per-step data to `observation_logs/` and computes reward-objective alignment metrics.

## Log files

For each run, we save:

- `step_log_YYYYMMDD_HHMMSS.json`
- `visit_frequency_YYYYMMDD_HHMMSS.json`
- `q_value_history_YYYYMMDD_HHMMSS.json`

## Alignment metrics

Computed on `state_type == "valid"` steps:

1. `Corr(R, MG)`
- Pearson correlation between normalized reward signal and true marginal gain.
- `R` is min-max normalized to `[0,1]`.

2. `HighR-LowMG%`
- Ratio of steps satisfying:
- `R > P90(R)` and `MG < P10(MG)`.

3. `MAE(|R_norm - MG|)`
- Point-wise mismatch between reward signal and marginal gain.

4. Bootstrap 95% CI
- Reported for correlation, HighR-LowMG, and MAE.

5. Coverage efficiency
- `States/100steps = UniqueStates / TotalSteps * 100`.

## Usage

Run summary + figures after a run:

```bash
python -c "from observation.analyzer import print_summary_statistics, generate_motivation_figures; print_summary_statistics('observation_logs'); generate_motivation_figures('observation_logs', 'motivation_figures.pdf')"
```

## Full-history marginal gain (recommended for motivation runs)

```powershell
$env:OBSERVATION_MG_FULL_HISTORY="1"
```

This uses full history for marginal gain computation (no sampling cap).
