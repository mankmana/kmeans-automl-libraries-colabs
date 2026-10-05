# AutoGluon: AutoML Capabilities Tour

This project demonstrates the capabilities of AutoGluon through an executed Google Colab notebook. The notebook covers tabular machine learning, time-series forecasting, multimodal learning, embeddings, interpretability, foundation models, and deployment.

## Links

- **Executed Google Colab:** [02 - AutoGluon executed.ipynb](https://colab.research.google.com/drive/1j89SkOgsdSG3386FZQaxBTBGc_44Ndor?usp=sharing)
- **Video tutorial:** [AutoGluon Colab Walkthrough](https://youtu.be/KCRkMpYmlKc)

## Assignment Coverage

This project covers the AutoGluon portions of the assignment:

- **Part 2:** AutoGluon landscape of capabilities
- **Part 3:** End-to-end AutoGluon machine learning with metrics and evaluation

The notebook is executed and includes outputs, visualizations, model comparisons, evaluation metrics, and deployment examples.

## What is AutoGluon?

AutoGluon is an AutoML framework that automatically trains, evaluates, and ensembles machine-learning models. It supports several data types and problem types while requiring relatively little model-selection code.

The notebook demonstrates the common AutoGluon workflow:

```python
predictor = SomePredictor(label="target").fit(
    train_data,
    time_limit=60
)

leaderboard = predictor.leaderboard(test_data)
predictions = predictor.predict(new_data)
