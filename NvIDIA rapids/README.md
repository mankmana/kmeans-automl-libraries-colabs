# NVIDIA RAPIDS: From Zero to Hero

This project demonstrates GPU-accelerated data science in Google Colab using NVIDIA RAPIDS. The notebook uses a free NVIDIA Tesla T4 GPU and covers cuDF, cuML, cuGraph, XGBoost, and related GPU data-science tools.

## Colab Notebook

[Open the executed Google Colab notebook](https://colab.research.google.com/drive/1_uCxWMqQzYAtjqT0pWUAbrSiSFSyQNWz?usp=sharing)

## Overview

The notebook presents a progressive, beginner-friendly walkthrough of GPU data science. It compares CPU and GPU workflows, analyzes a household-budget dataset, trains machine-learning models, and demonstrates how RAPIDS can accelerate common data-science tasks.

## Topics Covered

- GPU computing concepts, parallelism, memory bandwidth, and columnar data layouts
- Google Colab GPU setup and RAPIDS environment verification
- cuDF as a GPU-accelerated alternative to pandas
- `cudf.pandas` for GPU acceleration with familiar pandas code
- Data analysis and visualization using GPU-backed dataframes
- cuML classification, regression, preprocessing, and evaluation metrics
- K-means, DBSCAN, PCA, UMAP, t-SNE, and nearest-neighbor workflows
- Cross-validation, model selection, and `cuml.accel`
- GPU memory management with RMM and scaling with Dask-cuDF
- GPU-accelerated XGBoost and SHAP model interpretation
- cuGraph network analysis
- Model saving, serving, CPU fallbacks, monitoring, and error analysis
- CPU-versus-GPU performance considerations and practical trade-offs

## Dataset and Prediction Tasks

The notebook creates and analyzes a household-budget dataset containing household-month records and monthly spending histories. The machine-learning examples focus on:

- Predicting whether a household will overspend next month
- Predicting next-month spending
- Exploring monthly spending patterns as time series
- Building relationships between households and merchants for graph analysis

## RAPIDS Components

| Component | Purpose |
| --- | --- |
| cuDF | GPU-accelerated dataframe operations |
| cuML | GPU-accelerated machine learning |
| cuGraph | GPU-accelerated graph analytics |
| XGBoost | Gradient-boosted models with GPU support |
| CuPy | GPU-compatible numerical array operations |
| RMM | GPU memory-pool management |
| Dask-cuDF | Distributed GPU dataframe processing |

## Execution Environment

The notebook was executed in Google Colab with an NVIDIA Tesla T4 GPU. It reports the installed GPU and RAPIDS versions during execution and includes fallback logic for environments where GPU libraries are unavailable.

## How to Run

1. Open the Colab notebook using the link above.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** when available.
3. Run the setup and installation cells if the environment requires them.
4. Run the notebook from top to bottom so that data, models, benchmarks, and visualizations are created in sequence.
5. Review the printed metrics and CPU/GPU timing comparisons.

## Key Takeaways

- GPUs are especially useful for large, parallel, batch-oriented workloads.
- GPU acceleration includes data-transfer and startup overhead, so small tasks may not be faster.
- Keeping data on the GPU and avoiding unnecessary CPU/GPU transfers improves performance.
- Benchmarks should include synchronization and warm-up runs for meaningful results.
- CPU baselines, fallback paths, memory monitoring, and error analysis are important for reliable deployment.

## Repository Contents

This repository contains the executed Colab notebook and supporting documentation for the NVIDIA RAPIDS demonstration.

