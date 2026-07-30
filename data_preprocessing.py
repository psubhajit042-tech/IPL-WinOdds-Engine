# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Data Preprocessing Module
Author: SUBHAJIT

This module handles loading, cleaning, and feature engineering for the 
IPL matches and deliveries datasets to produce a dataset optimized for training.
"""

import os
import numpy as np
import pandas as pd

# The 8 standard teams used in the prediction model
TEAMS = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

def load_data(matches_path='matches.csv', deliveries_path='deliveries.csv'):
    """
    Loads matches and deliveries datasets from CSV files.
    """
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Matches CSV file not found at: {os.path.abspath(matches_path)}")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Deliveries CSV file not found at: {os.path.abspath(deliveries_path)}")
        
    print(f"Loading datasets...")
    matches_df = pd.read_csv(matches_path)
    deliveries_df = pd.read_csv(deliveries_path)
    
    print(f"Matches shape: {matches_df.shape}")
    print(f"Deliveries shape: {deliveries_df.shape}")
    return matches_df, deliveries_df

def preprocess_data(matches_df, deliveries_df):
    """
    Cleans and performs feature engineering on raw IPL matches and deliveries data.
    
    Returns:
        final_df (pd.DataFrame): Shuffled and cleaned dataframe ready for modeling.
        delivery_df (pd.DataFrame): Detailed second innings data (useful for progression analysis).
    """
    print("Preprocessing matches and deliveries data...")
    
    # 1. Calculate total score in the 1st inning of each match
    total_score_df = deliveries_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]
    
    # 2. Merge 1st inning totals back to matches dataframe
    match_df = matches_df.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')
    
    # 3. Standardize and clean team names
    # Mapping old franchise names to current ones
    team_replacements = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    
    for old_team, new_team in team_replacements.items():
        match_df['team1'] = match_df['team1'].str.replace(old_team, new_team)
        match_df['team2'] = match_df['team2'].str.replace(old_team, new_team)
        
    # 4. Filter only standard 8 teams
    match_df = match_df[match_df['team1'].isin(TEAMS)]
    match_df = match_df[match_df['team2'].isin(TEAMS)]
    
    # 5. Keep relevant match information
    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
    
    # 6. Merge matches with deliveries to look at ball-by-ball events
    delivery_df = match_df.merge(deliveries_df, on='match_id')
    
    # 7. Restrict model to the second innings (chasing team runs)
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    
    # Ensure run columns are numeric
    delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')
    
    # 8. Feature Engineering:
    # A. Current score in second innings
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
    
    # B. Runs remaining to chase the target (total_runs_x represents 1st innings score + 1, wait, actually total_runs_x is target)
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    
    # C. Balls remaining out of 120 (taking into account the current over and ball)
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
    
    # D. Wickets remaining (max 10)
    delivery_df['player_dismissed'] = np.where(delivery_df['player_dismissed'].notna(), 1, 0)
    delivery_df['wickets_fallen'] = delivery_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
    delivery_df['wickets_fallen'] = delivery_df['wickets_fallen'].clip(upper=10)
    delivery_df['wickets'] = 10 - delivery_df['wickets_fallen']
    
    # E. Run Rates: Current Run Rate (CRR) and Required Run Rate (RRR)
    # Avoid dividing by zero for CRR when 0 balls are bowled
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = np.where(
        delivery_df['balls_left'] == 0, 
        0, 
        delivery_df['runs_left'] / (delivery_df['balls_left'] / 6)
    )
    
    # F. Ground Truth: Did the chasing team win? (result = 1 if yes, else 0)
    def result_indicator(row):
        return 1 if row['batting_team'] == row['winner'] else 0
        
    delivery_df['result'] = delivery_df.apply(result_indicator, axis=1)
    
    # 9. Extract model training features
    # NOTE: 'match_id' is intentionally retained as the FIRST column so that
    # model_training.py can perform GROUP-AWARE splitting (GroupShuffleSplit /
    # GroupKFold by match_id). It is NOT a predictive feature: it is stripped
    # before any model sees the data. Keeping balls from the same match out of
    # both train and test prevents the random-row split from leaking match
    # trajectories (which previously inflated Random Forest to ~99.8%).
    feature_cols = [
        'match_id',                  # grouping key only, removed before training
        'batting_team', 'bowling_team', 'city', 'runs_left',
        'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result'
    ]
    final_df = delivery_df[feature_cols].copy()
    
    # 10. Shuffle and Clean
    # Shuffle dataframe rows (match_id grouping key must stay attached to its row)
    final_df = final_df.sample(final_df.shape[0], random_state=42)
    # Drop rows with NaN values (e.g. if city was empty)
    final_df.dropna(inplace=True)
    # Filter out records where balls_left is 0 to avoid inf RRR and divide-by-zero
    final_df = final_df[final_df['balls_left'] != 0]
    
    print(f"Preprocessed dataset final shape: {final_df.shape}")
    return final_df, delivery_df

if __name__ == '__main__':
    # Local execution demo
    try:
        matches_df, deliveries_df = load_data()
        final_df, _ = preprocess_data(matches_df, deliveries_df)
        final_df.to_csv('cleaned_data.csv', index=False)
        print("Data preprocessed successfully! Cleaned data saved as 'cleaned_data.csv'.")
    except Exception as e:
        print(f"An error occurred during local execution: {e}")
        print("Ensure 'matches.csv' and 'deliveries.csv' are located in this directory.")
