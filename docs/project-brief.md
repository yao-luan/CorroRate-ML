# CorroRate-ML Project Brief

## One-line Summary

This project builds a reproducible materials informatics baseline for predicting corrosion rate from alloy composition and atmospheric exposure features.

## What the Project Demonstrates

- Turning a course project into a clearer research codebase with documented data, methods, results and limitations.
- Understanding the full tabular ML workflow: preprocessing, train/test separation, feature selection, model comparison, evaluation and interpretation.
- Reasoning about leakage, reproducibility and model auditability rather than only reporting a final score.
- Preserving exploratory notebooks while also extracting a reusable command-line experiment.

## Technical Core

The pipeline starts from an Excel dataset containing 89 samples, 21 predictors and the target variable `RATE`. Predictors include alloy composition and environmental variables such as temperature, humidity, wetting time, precipitation, chloride and sulfur dioxide indicators.

The modeling workflow is:

1. Split data into training and test sets.
2. Fit scalers on training data only.
3. Rank features with mutual information and random forest importance.
4. Select the top features using agreement between the two ranking methods.
5. Train Random Forest and Ridge Regression baselines.
6. Compare RMSE, MAE and R2 on the held-out test set.
7. Generate prediction plots, feature importance plots and optional SHAP analysis.

## Why It Is Research-relevant

Corrosion datasets are often small, noisy and domain-dependent. A strong baseline is therefore not just a model, but a controlled experiment that makes assumptions visible. This project emphasizes:

- careful train/test separation;
- interpretable feature screening;
- comparison between nonlinear and linear baselines;
- saved artifacts for result inspection;
- explicit limitations and future validation needs.

## Current Limitations

- The dataset is small, so the result should be treated as exploratory.
- The current split is a random hold-out split; future work should evaluate environment-aware or time-aware validation.
- The baseline does not yet include uncertainty estimation.
- External validation is needed before making domain-level claims.

## Next Research Directions

- Add leave-one-site or leave-one-environment-out validation if metadata is available.
- Compare tree ensembles with Gaussian Process Regression or Bayesian models for uncertainty-aware prediction.
- Add permutation importance and stability analysis across repeated splits.
- Expand the dataset and test whether selected features remain stable.
- Package the notebook workflow into smaller reusable modules and add automated smoke tests.
