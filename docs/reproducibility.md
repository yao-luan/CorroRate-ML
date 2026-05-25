# Reproducibility

This document records the reproducibility assumptions used by the command-line baseline.

## Default Command

```bash
python scripts/run_experiment.py
```

Equivalent installed command:

```bash
corrorate-ml
```

## Default Configuration

| Option | Default |
| --- | --- |
| Data file | `datasets/project_target_dataset.xlsx` |
| Sheet | `10year` |
| Target | `RATE` |
| Test size | `0.2` |
| Random seed | `42` |
| Selected features | `10` |
| Cross-validation folds | `5` |
| Output directory | `outputs/corrosion_baseline/` |

## Leakage Controls

- The train/test split is created before scaling.
- `MinMaxScaler` is fitted on the training split only.
- Feature ranking is computed on the training split only.
- Hyperparameter search is performed inside cross-validation on the training split.
- Test data is used only for final held-out evaluation.

## Generated Artifacts

The baseline writes generated files to `outputs/corrosion_baseline/`:

- `metrics.csv`
- `predictions.csv`
- `selected_features.csv`
- `performance_scatter.png`
- `best_model_features.png`
- `shap_summary.png` and `shap_bar.png`, if SHAP runs successfully
- `run_config.json`

The `outputs/` folder is intentionally ignored by git because these files are reproducible artifacts.

## Notes

SHAP can be sensitive to library versions. If SHAP plotting fails, the core metrics and feature-importance artifacts are still generated. To skip SHAP explicitly:

```bash
python scripts/run_experiment.py --skip-shap
```
