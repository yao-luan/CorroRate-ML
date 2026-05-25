# CorroRate-ML

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](requirements.txt)
[![Reproducible](https://img.shields.io/badge/Experiment-Reproducible-brightgreen.svg)](scripts/run_experiment.py)
[![Notebook](https://img.shields.io/badge/Workflow-Jupyter-lightgrey.svg)](course_project/project.ipynb)

**A Reproducible Materials Informatics Baseline for Corrosion Rate Prediction**  
**面向材料腐蚀速率预测的可复现材料信息学基线**

一个面向**材料腐蚀速率预测**的可复现材料信息学项目。项目源自北京科技大学《材料数据挖掘利用方法》课程设计，并进一步整理为包含数据、Notebook、命令行实验、结果图、项目 brief 和开源规范的研究型 baseline。

> **Research question**：给定合金元素成分与大气暴露环境特征，能否构建一个可复现、可解释的腐蚀速率预测流程？

## 30-second Overview

| Item | Description |
| --- | --- |
| Task | Corrosion-rate prediction from material composition and exposure environment |
| Dataset | `datasets/project_target_dataset.xlsx`, sheet `10year` |
| Scale | 89 samples, 21 predictors, target `RATE` |
| Methods | Mutual information, random forest feature importance, Random Forest, Ridge Regression, SHAP |
| Reproduction | `python scripts/run_experiment.py` |
| Default best model | Random Forest Regressor |
| Default test result | RMSE `0.0220`, MAE `0.0134`, R2 `0.8218` |

The default result is produced with `random_state=42` and a held-out 20% test split. Because the dataset is small, these numbers should be treated as a reproducible baseline, not as a universal corrosion-performance claim.

## Why This Project Is More Than a Course Archive

- **Research workflow**: turns a course project into a clear experimental pipeline with data, assumptions, metrics and limitations.
- **Reproducibility**: extracts the notebook workflow into a command-line experiment that regenerates metrics, predictions and figures.
- **Leakage awareness**: performs train/test split before scaling; fits scalers and feature selection on the training set only.
- **Model comparison**: evaluates a nonlinear ensemble baseline and a linear regularized baseline under the same protocol.
- **Interpretability**: uses feature importance and SHAP-style explanations to inspect which variables drive predictions.
- **Auditability**: keeps notebooks, reports, result snapshots and generated artifacts aligned with the code.

## Quick Links for Reviewers

- [Project brief](docs/project-brief.md): concise research-facing summary.
- [Reproducibility notes](docs/reproducibility.md): experiment assumptions and generated artifacts.
- [Main experiment script](scripts/run_experiment.py): one-command reproduction entry point.
- [Reusable experiment module](src/corrorate_ml/experiment.py): data loading, feature selection, modeling and evaluation.
- [Core notebook](course_project/project.ipynb): original exploratory workflow.
- [Result figures](result/): saved visual evidence from the course project and baseline analysis.

## Method Pipeline

```text
Excel dataset
  -> numeric feature extraction
  -> train/test split
  -> train-only MinMax scaling
  -> mutual information ranking
  -> random forest importance ranking
  -> top-k feature selection
  -> Random Forest + Ridge Regression
  -> RMSE / MAE / R2 evaluation
  -> prediction plots + feature importance + optional SHAP
```

Default selected features from the command-line baseline:

```text
TOW, T_MAX, WIND_AVE, Cl, ULTRA, SUN, SOLAR, PRECIPIT, WIND_MAX, T_MIN
```

These variables cover exposure duration, temperature, wind, chloride, ultraviolet radiation, sunshine and precipitation, which are physically meaningful for atmospheric corrosion analysis.

## Baseline Results

Default command:

```bash
python scripts/run_experiment.py
```

Default metrics:

| Model | Train RMSE | Train R2 | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.0200 | 0.8470 | 0.0220 | 0.0134 | 0.8218 |
| Ridge Regression | 0.0201 | 0.8460 | 0.0224 | 0.0137 | 0.8163 |

The result suggests that, under the current split and feature-selection protocol, both models capture useful structure in the dataset, with Random Forest providing a slightly stronger held-out baseline.

## Installation

```bash
git clone https://github.com/yao-luan/CorroRate-ML.git
cd CorroRate-ML
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_experiment.py
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py
```

Install as an editable local package:

```bash
pip install -e .
corrorate-ml
```

Some coursework notebooks use TensorFlow/Keras. The main corrosion baseline does not require deep learning. To run those optional notebooks:

```bash
pip install -r requirements-deeplearning.txt
```

## Generated Artifacts

Running the baseline writes reproducible artifacts to `outputs/corrosion_baseline/`:

| File | Purpose |
| --- | --- |
| `metrics.csv` | model comparison with RMSE, MAE and R2 |
| `selected_features.csv` | feature rankings and final selected features |
| `predictions.csv` | held-out true values and model predictions |
| `performance_scatter.png` | predicted vs observed corrosion rate |
| `best_model_features.png` | feature importance or coefficients for the best model |
| `shap_summary.png` / `shap_bar.png` | optional SHAP figures |
| `run_config.json` | experiment configuration |

`outputs/` is ignored by git because these files are generated from the tracked data and scripts.

## Result Gallery

Selected visual outputs from [`result/`](result/):

<p align="center">
  <img src="result/25模型性能结果汇总.png" width="48%" alt="Model performance summary">
  <img src="result/11最佳模型随机森林模型特征重要性.png" width="48%" alt="Random forest feature importance">
</p>

<p align="center">
  <img src="result/18SHAP特征重要性摘要图-随机森林.png" width="48%" alt="SHAP feature importance summary">
  <img src="result/21特征相关性热力图.png" width="48%" alt="Feature correlation heatmap">
</p>

## Repository Structure

```text
.
├─ src/corrorate_ml/               # Reproducible experiment package
├─ scripts/run_experiment.py       # Command-line entry point
├─ course_project/                 # Core notebook, report and slides
├─ assignments/                    # Course assignments and practice notebooks
├─ datasets/                       # Dataset files and descriptions
├─ result/                         # Saved figures and result snapshots
├─ docs/                           # Project brief and reproducibility notes
├─ requirements.txt
├─ requirements-deeplearning.txt
├─ pyproject.toml
├─ CITATION.cff
├─ CONTRIBUTING.md
└─ LICENSE
```

## Project Positioning

This repository is best described as a **materials informatics research baseline** rather than a production software package. It is designed to show:

- how a materials dataset is converted into a modeling problem;
- how feature-selection and model-evaluation decisions are made explicit;
- how exploratory notebooks can be preserved while a reproducible script is extracted;
- how model performance and interpretability can be reported with appropriate caution.

## Limitations and Next Steps

Current limitations:

- The dataset is small, so conclusions are exploratory.
- The default validation is a random hold-out split; environment-aware or time-aware validation would be stronger.
- External validation is needed before making broader corrosion-model claims.
- Uncertainty estimation is not yet included.

Planned improvements:

- [x] Organize coursework, datasets, notebooks and result figures.
- [x] Add open-source license, citation metadata and contribution guide.
- [x] Add reproducible command-line experiment pipeline.
- [ ] Refactor more notebook logic into reusable modules.
- [ ] Add automated smoke tests for data loading and model training.
- [ ] Add repeated-split or leave-one-environment-out validation.
- [ ] Add uncertainty-aware prediction or confidence intervals.

## Citation

If this repository helps your study or research, please cite it using [`CITATION.cff`](CITATION.cff).

## License

Code and original documentation are released under the [MIT License](LICENSE). Third-party course materials or datasets should be used according to their original source and context.
