# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Model Training Module
Author: SUBHAJIT

This module builds the scikit-learn pipeline (featuring OneHotEncoding ColumnTransformer),
handles training/testing data cleaning, runs robust Stratified 10-Fold Cross-Validation,
prints evaluation metrics, and serializes the trained pipelines to pickle files.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Suppress the sklearn unknown-categories warning during group-aware CV folds.
# This fires when a test fold contains a city not seen during training (which is
# expected and correct behavior with handle_unknown='ignore').
warnings.filterwarnings('ignore', message='Found unknown categories')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

def get_column_transformer():
    """
    Returns a ColumnTransformer that applies One-Hot Encoding to the categorical features:
    'batting_team', 'bowling_team', 'city'. Remaining columns are passed through.
    """
    # handle_unknown='ignore' will ignore unknown categories in test/prediction sets
    # sparse_output=False ensures dense array outputs (required by some downstream steps)
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), 
         ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    return trf

def create_pipeline(classifier_type='logistic_regression'):
    """
    Creates and returns a scikit-learn Pipeline with preprocessing and classifier steps.

    Parameters:
        classifier_type (str): 'logistic_regression', 'random_forest', or 'xgboost'
    """
    trf = get_column_transformer()

    ct = classifier_type.lower()
    if ct == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', max_iter=1000)
    elif ct == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
    elif ct == 'xgboost':
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Run `pip install xgboost`.")
        # Tuned for moderate depth to avoid overfitting the (now group-aware) split.
        classifier = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'logistic_regression', 'random_forest', or 'xgboost'.")

    pipe = Pipeline(steps=[
        ('step1', trf),
        ('step2', classifier)
    ])

    return pipe

def clean_data_arrays(df):
    """
    Utility to replace infinite values with NaN and fill NaNs with 0.
    Ensures scikit-learn estimators do not encounter infinite/NaN values.
    """
    # Replace inf and -inf with NaN
    cleaned_df = df.replace([np.inf, -np.inf], np.nan)
    # Fill NaN values with 0
    cleaned_df = cleaned_df.fillna(0)
    return cleaned_df

def _clone_pipeline(pipeline):
    """
    Returns a fresh unfitted copy of a Pipeline (clone of each step).
    Used during group cross-validation so each fold trains from scratch.
    """
    from sklearn.base import clone
    return clone(pipeline)

def run_cross_validation(pipeline, X, y, groups, name="Model", n_splits=5):
    """
    Performs Group K-Fold Cross-Validation on the pipeline model, grouping by
    `match_id` so that NO match contributes balls to both train and test folds.
    This prevents the same-match trajectory leakage that previously inflated
    Random Forest / tree-based models to ~99.8%.

    Parameters:
        pipeline : scikit-learn Pipeline
        X        : feature DataFrame (must NOT include match_id)
        y        : target Series
        groups   : match_id Series aligned with X/y, used for grouping
        name     : display label
        n_splits : number of CV folds
    """
    X_clean = clean_data_arrays(X)

    # GroupKFold keeps every match_id entirely inside one fold.
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n--- {name} Group {n_splits}-Fold Cross-Validation (no same-match leakage) ---")
    scores = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_clean, y, groups), start=1):
        pipe_fold = _clone_pipeline(pipeline)
        pipe_fold.fit(X_clean.iloc[tr_idx], y.iloc[tr_idx])
        fold_acc = accuracy_score(y.iloc[te_idx], pipe_fold.predict(X_clean.iloc[te_idx]))
        scores.append(fold_acc)
        print(f"  Fold {fold}: acc={fold_acc:.4f}")
    scores = np.array(scores)
    print(f"Individual fold accuracies: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation of accuracy: {scores.std():.4f}")

    return scores

def evaluate_predictions(y_true, y_pred_class, y_pred_prob, model_name="Model"):
    """
    Calculates and prints classification and regression metrics, including
    ROC-AUC which measures how well the predicted win probability separates
    winners from losers (useful for a probability predictor).
    """
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class)
    rec = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    mae = mean_absolute_error(y_true, y_pred_prob)
    r2 = r2_score(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    print(f"\n--- {model_name} Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"MAE (Win Prob): {mae:.4f}")
    print(f"R2 (Win Prob):  {r2:.4f}")

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': auc,
        'MAE': mae,
        'R2': r2
    }

def train_and_evaluate_all(cleaned_data_path='cleaned_data.csv'):
    """
    Main driver function to load preprocessed data, split into train/test,
    clean feature matrices, train models, evaluate, and save models.

    Anti-leakage design
    -------------------
    The split is GROUP-AWARE: `GroupShuffleSplit` partitions by `match_id` so an
    entire match lives either in train or in test, never both. A naive random
    row split previously let balls from the same match land in both sets, which
    tree models memorized -> Random Forest reported an impossible ~99.8%.
    Grouping removes that leak; the honest accuracy ceiling here is ~78%.

    `match_id` is the first column of the cleaned data. It is stripped from X
    before any model sees it; it is only used as the grouping key for the split.
    """
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned data CSV file not found at: {cleaned_data_path}. Please run preprocessing first.")

    df = pd.read_csv(cleaned_data_path)

    # match_id is the grouping key (column 0). Separate it out, then X = features, y = result.
    if 'match_id' not in df.columns:
        raise ValueError(
            "Expected 'match_id' column in cleaned_data.csv for group-aware splitting. "
            "Re-run data_preprocessing.py to regenerate it."
        )
    groups = df['match_id']
    y = df['result']
    X = df.drop(columns=['match_id', 'result'])

    # Group-aware train/test split: whole matches stay together.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(f"Train rows: {len(X_train):,} ({groups_train.nunique()} matches) | "
          f"Test rows: {len(X_test):,} ({groups.iloc[test_idx].nunique()} matches) | "
          f"(no match appears in both)")

    # Clean feature matrices (replace inf/nan)
    X_train_clean = clean_data_arrays(X_train)
    X_test_clean = clean_data_arrays(X_test)
    X_all_clean = clean_data_arrays(X)

    # Initialize pipelines (XGBoost optional)
    pipes = {'Logistic Regression': create_pipeline('logistic_regression'),
             'Random Forest': create_pipeline('random_forest')}
    if _HAS_XGB:
        pipes['XGBoost'] = create_pipeline('xgboost')
    else:
        print("\n[warn] xgboost not installed -> skipping XGBoost. Run `pip install xgboost` to include it.")

    # 1. Fit each pipeline on the group-aware training split
    trained_metrics = {}
    cv_scores = {}
    for name, pipe in pipes.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train_clean, y_train)

        # 2. Group-aware Cross-Validation
        cv_scores[name] = run_cross_validation(
            pipe, X_all_clean, y, groups=groups, name=name, n_splits=5
        )

        # 3. Detailed Test Set Evaluation
        y_pred_class = pipe.predict(X_test_clean)
        y_pred_prob = pipe.predict_proba(X_test_clean)[:, 1]  # prob of win (class 1)
        trained_metrics[name] = evaluate_predictions(y_test, y_pred_class, y_pred_prob, name)

    # 4. Summary - which model generalizes best
    print("\n==================== TRAINING SUMMARY ====================")
    for name, scores in cv_scores.items():
        print(f"{name:20s} Mean CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_scores, key=lambda n: cv_scores[n].mean())
    print(f"\nConclusion: {best_name} generalizes best under group-aware CV "
          f"(~{cv_scores[best_name].mean()*100:.1f}% on unseen matches).")
    print("==========================================================")

    # 5. Compile comparison metrics into a DataFrame for visualization
    metrics_data = []
    for model_name, metrics in trained_metrics.items():
        for metric_name, val in metrics.items():
            # Keep classification metrics for the bar plot comparability
            if metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                metrics_data.append({'Model': model_name, 'Metric': metric_name, 'Score': val})
    metrics_df = pd.DataFrame(metrics_data)

    # Return pipelines keyed by name; keep LR/RF positions for backwards compatibility
    pipe_lr = pipes.get('Logistic Regression')
    pipe_rf = pipes.get('Random Forest')
    return pipe_lr, pipe_rf, metrics_df, pipes

def save_model(pipeline, filepath='model.pkl'):
    """
    Serializes a scikit-learn pipeline into a pickle file.
    """
    print(f"Saving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved successfully!")

if __name__ == '__main__':
    # Local execution demo
    try:
        pipe_lr, pipe_rf, _, all_pipes = train_and_evaluate_all()
        # Save every trained pipeline (lr, rf, and xgb if available)
        save_model(pipe_lr, 'pipe_lr.pkl')
        save_model(pipe_rf, 'pipe_rf.pkl')
        if 'XGBoost' in all_pipes:
            save_model(all_pipes['XGBoost'], 'pipe_xgb.pkl')
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please verify 'cleaned_data.csv' is generated by data_preprocessing.py first.")
