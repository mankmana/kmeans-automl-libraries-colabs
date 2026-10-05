# PyCaret: A Whirlwind Tour of Capabilities

This project demonstrates PyCaret's low-code machine-learning workflow across multiple supervised, unsupervised, time-series, explainability, and MLOps tasks. The notebook uses small generated industry datasets so that each capability can be executed and understood independently.

## Colab Notebook

[Open the executed Google Colab notebook](https://colab.research.google.com/drive/1V9lPItGIPw6QwmKyiWY8mbGXUC-2YTba?usp=sharing)

## Overview

The notebook follows a consistent PyCaret workflow:

```text
setup → compare_models → create/tune/blend/stack → plot_model → predict_model → finalize/save
```

Each section explains the business problem, required input format, important PyCaret calls, model outputs, evaluation metrics, and appropriate visualizations.

## Capabilities Covered

- Binary classification using a telecom customer-churn dataset
- Multiclass classification using loan-grade data
- Regression using house-price data
- Hyperparameter tuning, ensembling, blending, stacking, and calibration
- Imbalanced classification and rare-event fraud detection
- Customer segmentation with clustering
- Transaction outlier detection with anomaly models
- Time-series forecasting with exogenous variables
- Text feature engineering with TF-IDF and dimensionality reduction
- Model interpretability using SHAP and permutation importance
- Fairness checks and group-level evaluation
- Model finalization, saving, loading, API generation, and deployment concepts
- Experiment logging with MLflow when available
- Population Stability Index (PSI), drift monitoring, and retraining

## PyCaret Modules

| Module | Main Use |
| --- | --- |
| `ClassificationExperiment` | Binary and multiclass classification |
| `RegressionExperiment` | Numeric prediction |
| `ClusteringExperiment` | Unsupervised customer or group segmentation |
| `AnomalyExperiment` | Detection of unusual or suspicious records |
| `TSForecastingExperiment` | Time-series forecasting |

## Main Concepts Demonstrated

- `setup()` builds preprocessing, detects feature types, creates train/test partitions, and configures cross-validation.
- `compare_models()` evaluates multiple algorithms and returns the best model according to a selected metric.
- `create_model()` trains a selected algorithm.
- `tune_model()`, `blend_models()`, and `stack_models()` improve or combine models.
- `plot_model()` provides diagnostic visualizations such as ROC curves, precision-recall curves, residuals, and feature importance.
- `predict_model()` scores held-out or new data.
- `finalize_model()` refits the selected model before deployment.
- `save_model()` and `load_model()` preserve the full preprocessing-and-model pipeline.

## Evaluation and Monitoring

The notebook emphasizes choosing metrics that match the business problem:

- Classification: AUC, PR-AUC, recall, precision, F1, MCC, and calibration
- Regression: MAE, RMSE, median absolute error, MAPE, R², and bias
- Clustering: silhouette score, Calinski-Harabasz score, and Davies-Bouldin score
- Forecasting: MASE, RMSSE, and SMAPE
- Monitoring: PSI, score-distribution changes, flagged-record percentages, and retraining comparisons

## Environment

The notebook was verified with PyCaret 3.3.2 and Python 3.11. PyCaret 3.3.2 requires a compatible Python version, so use a Colab runtime based on Python 3.11 if installation or import errors occur.

The notebook runs on a Colab CPU and does not require an API key or external dataset download.

## How to Run

1. Open the Colab notebook using the link above.
2. Select a Python 3.11 Colab runtime if needed.
3. Run the installation cell for `pycaret[analysis,models]==3.3.2`.
4. Restart the runtime if Colab requests it, then run the notebook from top to bottom.
5. Review the generated datasets, model-comparison tables, metrics, plots, saved artifacts, and drift-monitoring results.

## Key Takeaways

- PyCaret provides a consistent low-code interface across several ML problem types.
- The `compare_models()` table is based on cross-validation; final performance should be checked on held-out data.
- Accuracy alone can be misleading for imbalanced problems, so metrics such as PR-AUC, recall, and MCC are important.
- A finalized model should be saved together with its library versions and evaluation report.
- Deployment requires monitoring input drift and model-score changes after release.

