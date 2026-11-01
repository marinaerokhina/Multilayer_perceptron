# Seoul Bike Sharing — Multilayer Perceptron Regression

[Русская версия](README_RUS.md) · [Lab 1: Linear Regression](https://github.com/marinaerokhina/Linear_regression)

**Python · TensorFlow/Keras · scikit-learn · Regression**

An educational comparison of four neural-network architectures for predicting hourly bicycle rental demand. The notebook includes a synthetic-data introduction, IQR-based outlier removal, model training, error metrics and visualizations.

## Data

The notebook starts with [SeoulBikeData_prepared.csv](SeoulBikeData_prepared.csv) from Lab 1 (8,756 rows). Further filtering of outliers using the interquartile range (IQR) leaves **8,371 rows** in [SeoulBikeData_prepared_no_outliers.csv](SeoulBikeData_prepared_no_outliers.csv). The filtering code is included, and the saved CSV is provided so it need not be regenerated.

Five features are used for M3/M4: **hour, temperature, rainfall, wind speed and snowfall**. M1/M2 use hour and temperature only. Inputs and target are normalized with `MinMaxScaler` and randomly split 70/30.

## Main architectures and results

The notebook's **saved execution** contains these results. M1–M3 were trained for 100 epochs; M4 was extended to 300 epochs.

| Model | Architecture | Epochs | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| M1 | 2 → 1 | 100 | 0.1924 | 0.413 |
| M2 | 2 → 64 → 32 → 1 | 100 | 0.1626 | 0.581 |
| M3 | 5 → 1 | 100 | 0.1858 | 0.453 |
| **M4** | **5 → 64 → 32 → 1** | **300** | **0.1441** | **0.671** |

The hidden layers use ReLU, output layers are linear, and models use Adam with MSE loss.

## Experiments with M4

The **original 100-epoch comparison**, recorded in a separate run, gave M4 **R² = 0.631, RMSE = 0.15257**. Increasing training to 300 epochs in the saved notebook gave **R² = 0.671, RMSE = 0.14410**.

Further *separately conducted* experiments:

| Hidden layers | Training | R² | RMSE | Parameters |
| --- | --- | ---: | ---: | ---: |
| 64–32 | 300 epochs | 0.671 | 0.14410 | 2,497 |
| 128–64 | 300 epochs | 0.673 | 0.14364 | 9,089 |
| 256–128 | 300 epochs | 0.675 | 0.14327 | 34,561 |
| 64–32 | 400 total epochs (continued training) | 0.684 | 0.14132 | 2,497 |

These are **distinct experiments**, not alternative metrics for one run. The 128–64, 256–128 and continued-400-epoch results are documented from earlier runs; their training outputs are not included in the notebook. Nearly **14×** more parameters for the largest architecture produced only a small improvement in R² (0.671 → 0.675) at 300 epochs.

**All RMSE values are measured on the normalized target scale**, not raw bike counts. Results can change with random initialization and software versions.

## Files and run

- [multilayer_perceptron.ipynb](multilayer_perceptron.ipynb) — executed notebook, source code, selected plots and analysis; large embedded teaching figures and verbose logs were omitted from the published copy for readability.
- [SeoulBikeData_prepared.csv](SeoulBikeData_prepared.csv) — output of Lab 1.
- [SeoulBikeData_prepared_no_outliers.csv](SeoulBikeData_prepared_no_outliers.csv) — IQR-filtered data used for neural-network training.
- [requirements.txt](requirements.txt) — Python dependencies.

Install the dependencies, open the notebook in Jupyter, and execute it from the repository directory.

## Limitations and dataset source

This is a coursework experiment, not a production prediction system. In the original notebook, scaling occurs **before** splitting into train/test, which may leak information. A chronological split would be preferable for forecasting future periods.

Dataset: [Seoul Bike Sharing Demand (UCI)](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), V. E. Sathishkumar, [Mendeley Data V2 (2020)](https://doi.org/10.17632/zbdtzxcxvg.2). This project uses a modified classroom dataset.
