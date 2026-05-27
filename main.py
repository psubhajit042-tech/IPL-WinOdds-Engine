# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Main Orchestrator
Author: SUBHAJIT

This script orchestrates the entire machine learning workflow:
1. Loads raw datasets (matches.csv, deliveries.csv)
2. Runs data preprocessing & feature engineering
3. Trains Logistic Regression & Random Forest models with 10-fold CV
4. Generates model comparison plots
5. Performs over-by-over match progression analysis on historical matches (981009 and 1237181)
6. Demonstrates real-time live situation prediction
"""

import os
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import train_and_evaluate_all, save_model
from predictor import load_trained_model, calculate_crr_rrr, predict_win_probability, match_progression
from visualization import plot_model_comparison, plot_match_progression

def main():
    print("==================================================================")
    print("                IPL WIN PREDICTOR - SYSTEM START                  ")
    print("==================================================================\n")
    
    # 1. Check for dataset existence
    matches_path = 'matches.csv'
    deliveries_path = 'deliveries.csv'
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        print(f"Error: Raw CSV files ('{matches_path}' and/or '{deliveries_path}') were not found in the current directory.")
        print(f"Directory contents: {os.listdir('.')}")
        return
        
    # 2. Data Preprocessing
    matches_df, deliveries_df = load_data(matches_path, deliveries_path)
    final_df, detailed_deliveries = preprocess_data(matches_df, deliveries_df)
    
    # Save cleaned data
    cleaned_csv_path = 'cleaned_data.csv'
    final_df.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned dataset saved successfully to '{cleaned_csv_path}'\n")
    
    # 3. Model Training & Cross-Validation
    pipe_lr, pipe_rf, metrics_df = train_and_evaluate_all(cleaned_csv_path)
    
    # Save both model pipelines to disk
    save_model(pipe_lr, 'pipe_lr.pkl')
    save_model(pipe_rf, 'pipe_rf.pkl')
    
    # 4. Model Comparison Visualization
    print("\nGenerating model performance comparison visualization...")
    plot_model_comparison(metrics_df, save_path='model_comparison.png')
    
    # 5. Load model and analyze historical match progressions
    print("\n--- Running Match Progression Analyses ---")
    model = load_trained_model('pipe_lr.pkl')
    
    # Match ID: 981009
    match_id_1 = 981009
    print(f"\nAnalyzing Match ID: {match_id_1}...")
    prog_df1, target1 = match_progression(detailed_deliveries, match_id_1, model)
    if prog_df1 is not None:
        print(prog_df1.head(10))
        plot_match_progression(prog_df1, target1, match_id=match_id_1, save_path='match_981009_progression.png')
        
    # Match ID: 1237181
    match_id_2 = 1237181
    print(f"\nAnalyzing Match ID: {match_id_2}...")
    prog_df2, target2 = match_progression(detailed_deliveries, match_id_2, model)
    if prog_df2 is not None:
        print(prog_df2.head(10))
        plot_match_progression(prog_df2, target2, match_id=match_id_2, save_path='match_1237181_progression.png')
        
    # 6. Run-time Prediction Scenario Demo
    print("\n--- Live Match Predictor Demonstration ---")
    # Live Scenario: Kolkata Knight Riders chasing 120 against Chennai Super Kings in Kolkata
    # Chasing team: KKR has scored 110 runs, faced 80 balls (40 balls remaining), and has 7 wickets in hand
    target_score = 120
    runs_scored = 110
    balls_faced = 80
    wickets_left = 7
    
    crr, rrr = calculate_crr_rrr(runs_scored, balls_faced, target_score)
    runs_remaining = target_score - runs_scored
    balls_remaining = 120 - balls_faced
    
    win_p, lose_p = predict_win_probability(
        model=model,
        batting_team='Kolkata Knight Riders',
        bowling_team='Chennai Super Kings',
        city='Kolkata',
        runs_left=runs_remaining,
        balls_left=balls_remaining,
        wickets=wickets_left,
        total_runs_x=target_score,
        crr=crr,
        rrr=rrr
    )
    
    print(f"\n🏏 Live Input Parameters:")
    print(f"   Batting Team:      Kolkata Knight Riders")
    print(f"   Bowling Team:      Chennai Super Kings")
    print(f"   City:              Kolkata")
    print(f"   Target:            {target_score} runs")
    print(f"   Current Score:     {runs_scored}/{10 - wickets_left} in {balls_faced // 6}.{balls_faced % 6} overs")
    print(f"   Current Run Rate:  {crr:.2f}")
    print(f"   Required Run Rate: {rrr:.2f}")
    print("--------------------------------------------------")
    print(f"🏏 Model Prediction:")
    print(f"   Kolkata Knight Riders Win Probability: {win_p:.2f}%")
    print(f"   Chennai Super Kings Win Probability:   {lose_p:.2f}%")
    print("==================================================\n")
    
if __name__ == '__main__':
    main()
