# Outlier Detection

This project compares the performance of several unsupervised outlier detection methods on real medical datasets and synthetic datasets. The main evaluation metric is AUC. The compared methods include KNN, LOF, COF, LDOF, HBOS, HDIOD, and the improved variants GATED_HDIOD and EG_HDIOD.

## Project Structure

```text
.
├── Data/                 # Real dataset files
├── results/              # Experimental outputs and plotting scripts
├── scripts/              # Experiment running scripts
└── src/                  # Data loading, algorithm implementation, and synthetic data generation
```

Main modules:

- `src/datasets.py`: Loads real and synthetic datasets in a unified way, including label conversion, feature encoding, and normalisation.
- `src/baselines.py`: Baseline methods including KNN, LOF, COF, LDOF, and HBOS.
- `src/hdiod.py`: Implementation of the original HDIOD method.
- `src/new_hdiod.py`: Fixed KNN score, base HDIOD, and GATED_HDIOD implementation.
- `src/eg_hdiod.py`: Implementation of EG_HDIOD.
- `src/synthetic.py`: Ten two-dimensional synthetic datasets for outlier detection.

## Environment Setup

Python 3.10 or above is recommended. If a `.venv` folder already exists in the project, the existing virtual environment can also be used.

Create and activate a new virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running Experiments

Please run the commands from the project root directory, which contains `src/`, `scripts/`, and `Data/`.

Run the full method comparison:

```powershell
python -m scripts.run_all
```

Run the traditional baseline methods:

```powershell
python -m scripts.run_baseline
```

Run HDIOD:

```powershell
python -m scripts.run_hdiod
```

Run the GATED_HDIOD comparison experiment:

```powershell
python -m scripts.run_new
```

Run the EG_HDIOD comparison experiment:

```powershell
python -m scripts.run_eg_compare
```

## Plots and Results

The result files are saved in the `results/` directory. Common output files include:

- `results/run_all_results.csv`
- `results/run_all_overall_leaderboard.csv`
- `results/run_all_per_dataset_leaderboards.csv`
- `results/run_new_results.csv`
- `results/run_new_overall_leaderboard.csv`
- `results/run_eg_compare_results.csv`
- `results/run_eg_compare_overall.csv`

Generate AUC-k curve plots:

```powershell
python -m results.plot_auc_k
```

Generate the synthetic dataset overview plot:

```powershell
python -m results.plot_synthetic_overview
```

## Datasets

The project includes the following real medical datasets:

- Breast Cancer
- heart_failure
- liver disorders
- Parkinsons
- Gallstone
- Hepatitis C Virus (HCV) for Egyptian patients
- Diabetic Retinopathy Debrecen
- Thoracic Surgery
- Cervical Cancer
- Cardiotocography

The project also includes ten synthetic datasets:

- `syn_gauss_uo`
- `syn_multi_blobs_uo`
- `syn_two_density`
- `syn_blocks`
- `syn_vshape`
- `syn_moons`
- `syn_spiral`
- `syn_double_spiral`
- `syn_sine`
- `syn_two_lines`

## Method Description

In this project, a larger outlier score means that a sample is more likely to be an outlier. The experiment scripts use `sklearn.metrics.roc_auc_score` to calculate the AUC of each method on each dataset.

The real dataset loading process is as follows:

1. Read the corresponding CSV file from the `Data/` directory.
2. Use the configuration in `src/datasets.py` to identify the label column and anomaly class.
3. Remove columns that are not used for modelling.
4. Apply one-hot encoding to non-numeric features.
5. Normalise the feature values to the range `[0, 1]`.

## Notes

- It is recommended to run scripts using `python -m scripts.xxx`. Do not run scripts directly inside the `scripts/` directory, otherwise the `src` package may not be imported correctly.
- Some historical encoding issues may exist in certain data files or comments, but they do not affect the main experimental logic.
- To add a new dataset, configure its file path, label column, anomaly label, and normalisation option in the `DATASETS` dictionary in `src/datasets.py`.
