# K-Means Clustering: From Zero to Hero

This repository contains an executed Google Colab notebook demonstrating K-means clustering, its mathematical foundations, practical variations, evaluation methods, failure cases, and modern applications.

## Notebook

Open the executed notebook in Google Colab:

[01 - k-means colab.ipynb](https://colab.research.google.com/drive/1DR1h4DMnH5qtGfWX31mbrcXrmLH9Oh2_?usp=sharing)
youtube Video : https://youtu.be/_DXfUvH6MpU

The notebook is designed to run on a free Colab CPU. It uses built-in or generated datasets and does not require API keys for the main exercises.

## Assignment Coverage

This notebook covers Part 1 of the assignment:

> Demonstrate K-means clustering and its variations.

The important code blocks are implemented from scratch where useful, executed, visualized, and compared with scikit-learn implementations.

## Main Topics

- What a cluster means: prototype-based, density-based, contiguity-based, and conceptual views
- K-means objective function and sum of squared errors, or SSE
- Why the SSE-optimal centroid is the arithmetic mean
- Euclidean, Manhattan, and cosine distance concepts
- Lloyd’s K-means algorithm implemented with NumPy
- Convergence and the non-increasing SSE property
- Random initialization and the K-means++ strategy
- Empty clusters, outliers, and local optima
- Bisecting K-means
- K-medians, K-medoids, spherical K-means, and mini-batch K-means
- Fuzzy c-means and Gaussian-mixture soft clustering
- Internal and external clustering metrics
- Choosing the number of clusters, `K`
- Random-data baselines and cluster tendency
- Feature scaling and common K-means failure cases
- Customer segmentation, image color quantization, and handwritten digits
- Explainability using centroid profiles and boundary points
- Saving and serving a scaler-plus-K-means pipeline
- Monitoring centroid shift and population-stability drift
- Embeddings, vector quantization, product quantization, and vector search
- GPU K-means and LLM-assisted cluster naming

## Datasets

The notebook uses several datasets to demonstrate both successful and unsuccessful use cases:

| Dataset | Purpose |
|---|---|
| `X_toy` | Small dataset for hand calculations and exercises |
| `X_blobs` | Clean, round clusters where K-means performs well |
| `X_moons` | Curved clusters showing a K-means failure case |
| Iris | Tabular data with known class labels for evaluation |
| Wine | Feature-scaling demonstration |
| Digits | Higher-dimensional image-style clustering example |

## Key Implementation Details

The notebook defines two reusable helper functions:

```python
plot_clusters(X, labels=None, centers=None)
sse(X, labels, centers)
