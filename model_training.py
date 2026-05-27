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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score

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
        classifier_type (str): 'logistic_regression' or 'random_forest'
    """
    trf = get_column_transformer()
    
    if classifier_type.lower() == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear')
    elif classifier_type.lower() == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'logistic_regression' or 'random_forest'.")
        
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

def run_cross_validation(pipeline, X, y, name="Model"):
    """
    Performs 10-Fold Stratified Cross-Validation on the pipeline model.
    Prints the individual, mean, and standard deviation accuracy scores.
    """
    # Clean the X data first
    X_clean = clean_data_arrays(X)
    
    # Define Stratified K-Fold
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    print(f"\n--- {name} Stratified 10-Fold Cross-Validation ---")
    scores = cross_val_score(pipeline, X_clean, y, cv=kf, scoring='accuracy')
    print(f"Individual fold accuracies: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation of accuracy: {scores.std():.4f}")
    
    return scores

def evaluate_predictions(y_true, y_pred_class, y_pred_prob, model_name="Model"):
    """
    Calculates and prints classification and regression metrics.
    """
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class)
    rec = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    mae = mean_absolute_error(y_true, y_pred_prob)
    r2 = r2_score(y_true, y_pred_prob)
    
    print(f"\n--- {model_name} Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MAE (Win Prob): {mae:.4f}")
    print(f"R2 (Win Prob):  {r2:.4f}")
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'MAE': mae,
        'R2': r2
    }

def train_and_evaluate_all(cleaned_data_path='cleaned_data.csv'):
    """
    Main driver function to load preprocessed data, split into train/test,
    clean feature matrices, train models, evaluate, and save models.
    """
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned data CSV file not found at: {cleaned_data_path}. Please run preprocessing first.")
        
    df = pd.read_csv(cleaned_data_path)
    
    # Split into features (X) and label (y)
    X = df.iloc[:, :-1]  # all columns except the last (result)
    y = df.iloc[:, -1]   # the 'result' column
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Clean train and test data matrices (replace inf/nan)
    X_train_clean = clean_data_arrays(X_train)
    X_test_clean = clean_data_arrays(X_test)
    X_all_clean = clean_data_arrays(X)
    
    # Initialize pipelines
    pipe_lr = create_pipeline('logistic_regression')
    pipe_rf = create_pipeline('random_forest')
    
    # 1. Fit Logistic Regression
    print("\nTraining Logistic Regression...")
    pipe_lr.fit(X_train_clean, y_train)
    
    # 2. Fit Random Forest Classifier
    print("Training Random Forest...")
    pipe_rf.fit(X_train_clean, y_train)
    
    # 3. Cross-Validation
    lr_cv_scores = run_cross_validation(pipe_lr, X_all_clean, y, name="Logistic Regression")
    rf_cv_scores = run_cross_validation(pipe_rf, X_all_clean, y, name="Random Forest")
    
    # 4. Detailed Test Set Evaluations
    # Predictions for Logistic Regression
    y_pred_class_lr = pipe_lr.predict(X_test_clean)
    y_pred_prob_lr = pipe_lr.predict_proba(X_test_clean)[:, 1] # prob of win (class 1)
    lr_metrics = evaluate_predictions(y_test, y_pred_class_lr, y_pred_prob_lr, "Logistic Regression")
    
    # Predictions for Random Forest
    y_pred_class_rf = pipe_rf.predict(X_test_clean)
    y_pred_prob_rf = pipe_rf.predict_proba(X_test_clean)[:, 1] # prob of win (class 1)
    rf_metrics = evaluate_predictions(y_test, y_pred_class_rf, y_pred_prob_rf, "Random Forest")
    
    # Summarize which model is better
    print("\n==================== TRAINING SUMMARY ====================")
    print(f"Logistic Regression Mean CV Accuracy: {lr_cv_scores.mean():.4f}")
    print(f"Random Forest Mean CV Accuracy:       {rf_cv_scores.mean():.4f}")
    
    if rf_cv_scores.mean() > lr_cv_scores.mean():
        print("Conclusion: Random Forest performs significantly better in accuracy.")
    else:
        print("Conclusion: Logistic Regression performs better/comparable in accuracy.")
    print("==========================================================")
    
    # Compile comparison metrics into a DataFrame for visualization
    metrics_data = []
    for model_name, metrics in [('Logistic Regression', lr_metrics), ('Random Forest', rf_metrics)]:
        for metric_name, val in metrics.items():
            # Filter only classification metrics for bar plot comparability
            if metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Score': val
                })
    metrics_df = pd.DataFrame(metrics_data)
    
    return pipe_lr, pipe_rf, metrics_df

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
        pipe_lr, pipe_rf, _ = train_and_evaluate_all()
        # Save the Logistic Regression pipeline (used commonly for smooth probabilities)
        save_model(pipe_lr, 'pipe_lr.pkl')
        # Save Random Forest pipeline as well
        save_model(pipe_rf, 'pipe_rf.pkl')
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please verify 'cleaned_data.csv' is generated by data_preprocessing.py first.")
