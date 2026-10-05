# AutoGluon End-to-End Machine Learning

This project demonstrates an end-to-end machine-learning workflow using AutoGluon. A household-budget dataset is used throughout the notebook to demonstrate classification, regression, time-series forecasting, multimodal learning, interpretability, deployment, and model monitoring.

## Links

- **Executed Google Colab:** [03 - AutoGluon end2end ML.ipynb](https://colab.research.google.com/drive/1iI_NUEFK9thw_1W4QPT781Rjg4e49yOv?usp=sharing)
- **Video tutorial:** [AutoGluon End-to-End ML Walkthrough](https://youtu.be/ui-v2dhc8jY)

## Assignment Coverage

This notebook covers the end-to-end AutoGluon machine-learning requirement, including model training, evaluation metrics, forecasting, multimodal data, interpretability, deployment, and MLOps concepts.

## Project Overview

The notebook uses a synthetic household-budget dataset containing household characteristics, income, spending categories, text notes, missing values, and monthly records.

The main prediction questions are:

1. Which households will overspend next month?
2. How much will each household spend next month?
3. What will each household’s future monthly spending look like?

The notebook emphasizes predicting the **next month** rather than the current month to avoid data leakage.

## Topics Demonstrated

- AutoML fundamentals and the AutoGluon workflow
- Baseline modeling with scikit-learn
- Binary classification using `TabularPredictor`
- Regression and quantile regression
- Model comparison with `leaderboard`
- Classification metrics including accuracy, F1, ROC-AUC, PR-AUC, log loss, and Brier score
- Regression metrics including MAE, RMSE, MAPE, R-squared, and bias
- Decision-threshold calibration
- Probability calibration
- Data leakage detection
- Presets, time limits, bagging, stacking, and hyperparameters
- Feature importance and model explanations
- Error analysis by data segment
- Time-series forecasting using `TimeSeriesPredictor`
- Prediction intervals and probabilistic forecasts
- Multimodal learning with text and tabular features
- Model persistence and deployment
- `refit_full`, `persist`, and `distill`
- Prediction-latency measurement
- Feature drift and prediction drift monitoring
- Foundation models and AutoGluon’s modern tabular model landscape

## Dataset Structure

The main dataset contains features such as:

- Household identifier
- City tier
- Household size
- Car ownership
- Income
- Rent
- Grocery spending
- Utility spending
- Transportation spending
- Dining and entertainment spending
- Free-text household notes
- Survey scores
- Referral codes
- Historical spending statistics

The main targets are:

- `overspent_next`: binary classification target
- `spend_next`: regression target
- Monthly spending values for time-series forecasting

## AutoGluon Workflow

The basic tabular workflow is:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label="overspent_next",
    eval_metric="roc_auc",
    path="model_directory"
).fit(
    train_data,
    time_limit=60,
    presets="medium_quality"
)

leaderboard = predictor.leaderboard(test_data)
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)
