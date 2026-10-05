# PyCaret: From Zero to Hero — MLOps

This project demonstrates an end-to-end, low-code machine-learning workflow with PyCaret. A household-budget dataset is used throughout the notebook to show classification, regression, forecasting, clustering, anomaly detection, model interpretation, deployment, and monitoring.

## Colab Notebook

[Open the executed Google Colab notebook](https://colab.research.google.com/drive/1SCx_T2qpI91BMSxf-8KzUnzf-60U6ODH?usp=sharing)

## Overview

The notebook follows the complete PyCaret lifecycle:

```text
setup → compare_models → create_model → tune_model → ensemble/blend/stack
→ calibrate/interpret → finalize_model → predict_model → save_model
```

It explains what each stage does, which defaults PyCaret applies, how to select appropriate metrics, and how to move a model from experimentation toward deployment.

## Dataset

The notebook generates a household-budget dataset with household-month records, income, spending categories, household information, free-text notes, missing values, and time-series history.

The main prediction tasks are:

- Predicting whether a household will overspend next month
- Predicting next month's total spending
- Forecasting monthly spending patterns
- Grouping households by spending behavior
- Detecting unusual household or transaction behavior

## Topics Covered

- PyCaret's low-code machine-learning lifecycle
- Deliberate preprocessing with imputation, encoding, scaling, transformations, feature selection, outlier handling, and class-imbalance options
- Classification model comparison and evaluation
- Model tuning, bagging, boosting, blending, stacking, calibration, and threshold optimization
- Regression, residual analysis, and leakage prevention
- SHAP-style model interpretation and reason codes
- Error analysis and segment-level diagnostics
- Clustering and anomaly detection
- Time-series forecasting with the `pycaret.time_series` module
- Model finalization, saving, loading, and prediction on new data
- API generation and deployment concepts
- Experiment logging with MLflow when available
- Data drift monitoring with Population Stability Index (PSI)
- Retraining after distribution shift
- GPU support through `use_gpu=True` and RAPIDS-compatible engines
- Using custom scikit-learn-compatible estimators and tabular foundation models
- Choosing between PyCaret, AutoGluon, RAPIDS, and scikit-learn

## PyCaret Modules

| Module | Purpose |
| --- | --- |
| `ClassificationExperiment` | Predict categorical outcomes such as overspending |
| `RegressionExperiment` | Predict numeric outcomes such as spending |
| `ClusteringExperiment` | Discover groups without a target column |
| `AnomalyExperiment` | Identify unusual observations |
| `TSForecastingExperiment` | Forecast future values in a time series |

## Evaluation Metrics

The notebook uses metrics appropriate to each task:

- Classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, log loss, and Brier score
- Regression: MAE, RMSE, median absolute error, MAPE, R², and bias
- Forecasting: MASE, RMSSE, and SMAPE
- Clustering: silhouette, Calinski-Harabasz, and Davies-Bouldin scores
- Anomaly detection: precision at a selected top-k threshold
- Monitoring: PSI, score-distribution changes, and flagged-record percentages

## Important Best Practices

- Use `test_data=` in `setup()` to preserve an honest held-out evaluation set.
- Do not interpret `compare_models()` cross-validation scores as final test scores.
- For imbalanced targets, use PR-AUC, recall, MCC, and a business-driven probability threshold instead of accuracy alone.
- Avoid leakage by predicting next month's outcome from information available today.
- Freeze the evaluation report before calling `finalize_model()`.
- Save the complete preprocessing-and-model pipeline together with library versions.
- Monitor input and score drift after deployment and retrain when the evidence supports it.

## Environment

The notebook was executed in Google Colab using PyCaret 3.3.2 and Python 3.11. It runs on a CPU without an API key or external dataset download.

## How to Run

1. Open the Colab notebook using the link above.
2. Use a Python 3.11 runtime if PyCaret installation fails on a newer Python version.
3. Run the installation and setup cells.
4. Restart the runtime if requested by Colab, then run the notebook from top to bottom.
5. Review the generated metrics, plots, saved artifacts, deployment examples, and drift-monitoring results.

## Key Takeaways

- PyCaret reduces repetitive machine-learning code while keeping the underlying pipeline inspectable.
- Good MLOps includes evaluation discipline, explainability, artifact versioning, deployment, logging, drift monitoring, and retraining.
- Tool selection should depend on data modality, row count, accuracy requirements, explainability, and serving constraints.

