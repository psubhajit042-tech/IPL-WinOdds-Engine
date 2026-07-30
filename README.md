# 🏏 IPL Win Predictor (Machine Learning Pipeline)

An end-to-end machine learning pipeline that predicts the win/loss probability of a chasing team in the Indian Premier League (IPL) in real-time. 

This repository has been fully modularized into professional, production-ready Python modules for clean presentation and ease of tracking.

---

## 📂 Project Structure

```text
ipl_win_Predictor/
│
├── matches.csv                  # Raw matches dataset
├── deliveries.csv               # Raw deliveries (ball-by-ball) dataset
│
├── data_preprocessing.py        # Data cleaning, team renames, and feature engineering
├── model_training.py            # Pipelines, cleaning matrices, training, and 10-fold CV
├── predictor.py                 # Live query inputs and historical match progression analytics
├── visualization.py             # Custom graphing helpers (Seaborn & Matplotlib)
├── main.py                      # Orchestrator script running the entire pipeline end-to-end
│
├── cleaned_data.csv             # Generated intermediate dataset (includes match_id grouping key)
├── pipe_lr.pkl                  # Serialized Logistic Regression model pipeline
├── pipe_rf.pkl                  # Serialized Random Forest model pipeline
├── pipe_xgb.pkl                 # Serialized XGBoost model pipeline
│
├── model_comparison.png         # Saved metric comparison chart
├── match_981009_progression.png # Progression plot for Match ID 981009
├── match_1237181_progression.png# Progression plot for Match ID 1237181
│
├── requirements.txt             # Project library dependencies
└── README.md                    # Project documentation (this file)
```

---

## ✨ Features

- **Robust Data Cleaning**: Automatically handles team renames (e.g. *Deccan Chargers* $\rightarrow$ *Sunrisers Hyderabad*, *Delhi Daredevils* $\rightarrow$ *Delhi Capitals*) and restricts analyses to the second innings.
- **Advanced Feature Engineering**: Calculates critical cricket metrics on a ball-by-ball basis:
  - Current Run Rate (CRR)
  - Required Run Rate (RRR)
  - Remaining Wickets & Remaining Balls
- **Anti-Leakage (Group-Aware) Splitting**: Train/test split and cross-validation are grouped by `match_id`, so every ball of a match stays entirely in train *or* test. This eliminated a same-match trajectory leak that previously inflated tree models to ~99.8%.
- **Triple Pipeline Architecture**: Compares `Logistic Regression`, `Random Forest`, and `XGBoost`, each combining `OneHotEncoder` preprocessing and classifier into a single pipeline.
- **Group-Aware Cross-Validation**: Evaluates all three classifiers with `GroupKFold` for honest, leakage-free stability across folds.
- **Match Win Progression Dashboard**: Plots over-by-over charts depicting runs scored, wickets lost, and fluctuating win/lose probability curves for historical matches.
- **Interactive Live Predictor**: Accepts manual match states and runs real-time predictions.

---

## ⚡ Prerequisites & Installation

A GPU is **not** required. The pipeline is optimized to run on standard CPUs in a matter of seconds.

1. **Activate your virtual environment** and navigate to the project directory:
   ```powershell
   cd "C:\Users\SUBHAJIT\Desktop\machine learning assignments\ipl_win_Predictor"
   ```

2. **Install the dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Installs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`)*

---

## 🚀 How to Run

You can choose to run the entire pipeline end-to-end at once, or run it step-by-step to track the progress.

### Option 1: Run the End-to-End Orchestrator (At Once)
Execute the main script to run all modules sequentially and output everything (trained models, cleaned CSVs, progression tables, and PNG plots):
```powershell
python main.py
```

### Option 2: Step-by-Step Execution (For Tracking)

#### Step 1: Preprocess Raw Data
Runs data cleaning and feature engineering.
```powershell
python data_preprocessing.py
```
* **Output**: Creates `cleaned_data.csv`.

#### Step 2: Train & Evaluate Models
Reads the cleaned CSV, runs Stratified K-Fold CV, evaluates accuracy/precision/recall metrics, and serializes the trained models.
```powershell
python model_training.py
```
* **Output**: Generates serialized models `pipe_lr.pkl` and `pipe_rf.pkl`.

#### Step 3: Run Prediction Demonstrations
Runs a real-time prediction using the loaded Logistic Regression model on a custom scenario.
```powershell
python predictor.py
```
* **Output**: Prints the live scenario predictions in the terminal.

#### Step 4: Verify Visualization Tools
Tests the Seaborn/Matplotlib visualization helpers using mock data.
```powershell
python visualization.py
```
* **Output**: Pops up mock comparison and mock progression plots on your screen and saves them.

---

## 📊 Model Evaluation Summary

Evaluated with **group-aware cross-validation** (`GroupKFold` by `match_id`, plus a `GroupShuffleSplit` holdout) so that no match contributes balls to both train and test. This prevents the same-match trajectory leakage that previously inflated tree models to ~99.8%.

| Model | Mean CV Accuracy | Std. Dev. | ROC-AUC (holdout) | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `77.29%` | `0.0269` | `0.8276` | Smooth, realistic, and continuous live probabilities |
| **Random Forest** | `75.62%` | `0.0157` | `0.8230` | Strong non-linear baseline |
| **XGBoost** | `76.77%` | `0.0237` | `0.8325` | Best probability calibration (lowest MAE) |

> **Why Logistic Regression wins here:** Under an honest (group-aware) split, linear boundaries on run-rate features generalize slightly better than tree ensembles. The earlier ~99.8% Random Forest figure was a data-leakage artifact of a random row split; this README now reports the defensible numbers.

---

## 📈 Visualizations Generated
When running the full pipeline, the following PNG files are automatically generated:

1. **`model_comparison.png`**: A professional bar chart comparing accuracy, precision, recall, and F1 scores of the trained models.
2. **`match_981009_progression.png` / `match_1237181_progression.png`**: Beautiful, descriptive over-by-over progression dashboards tracking wickets, runs per over, and the live win/lose percentage curves.