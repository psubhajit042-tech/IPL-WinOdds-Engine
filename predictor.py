# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Prediction Module
Author: SUBHAJIT

This module provides real-time and historical prediction utilities.
It houses functions to compute run rates, predict win probabilities for live scenarios,
and analyze the over-by-over probability progression of historic matches.
"""

import pickle
import numpy as np
import pandas as pd

def load_trained_model(filepath='pipe_lr.pkl'):
    """
    Loads a saved scikit-learn pipeline from a pickle file.
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Trained model not found at {filepath}. Please train and save the model first.")

def calculate_crr_rrr(runs_scored, balls_faced, target):
    """
    Helper function to calculate Current Run Rate (CRR) and Required Run Rate (RRR)
    based on match state.
    
    Parameters:
        runs_scored (int): Total runs scored by the batting team so far in 2nd innings.
        balls_faced (int): Total legal balls faced so far in 2nd innings.
        target (int): Target runs to win (1st innings total score + 1).
        
    Returns:
        crr (float): Current Run Rate.
        rrr (float): Required Run Rate.
    """
    overs_faced = balls_faced / 6
    runs_remaining = target - runs_scored
    overs_remaining = 20 - overs_faced  # Standard T20 is 20 overs

    crr = runs_scored / overs_faced if overs_faced > 0 else 0
    rrr = runs_remaining / overs_remaining if overs_remaining > 0 else 0

    return round(crr, 2), round(rrr, 2)

def predict_win_probability(model, batting_team, bowling_team, city, runs_left, balls_left, wickets, total_runs_x, crr, rrr):
    """
    Predicts the win/loss probability of a single game situation.
    
    Returns:
        win_prob (float): Percentage probability of the batting team winning (0-100).
        lose_prob (float): Percentage probability of the bowling team winning (0-100).
    """
    # Create input DataFrame with identical structure as training feature matrix X
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [total_runs_x],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    # Predict probabilities
    pred_proba = model.predict_proba(input_df)
    
    # Extract probabilities (0 is Lose/Bowling team win, 1 is Win/Batting team win)
    lose_prob = np.round(pred_proba[0][0] * 100, 2)
    win_prob = np.round(pred_proba[0][1] * 100, 2)
    
    return win_prob, lose_prob

def match_progression(deliveries_df, match_id, model):
    """
    Computes the over-by-over win/loss progression for a specific historical match.
    Only analyzes the end of each over (ball == 6).
    
    Parameters:
        deliveries_df (pd.DataFrame): The full deliveries dataset (second innings).
        match_id (int): Unique ID of the match to analyze.
        model: Trained scikit-learn pipeline.
        
    Returns:
        progression_df (pd.DataFrame): Dataframe with over-by-over progression (lose%, win%, runs, wickets).
        target (int): Target runs for the match.
    """
    # 1. Filter rows for the selected match and get end-of-over events
    match = deliveries_df[deliveries_df['match_id'] == match_id]
    match = match[match['ball'] == 6].copy()

    if match.empty:
        print(f"No ball-by-ball end-of-over (ball == 6) data found for match_id: {match_id}")
        return None, None

    # 2. Select only model feature columns
    feature_cols = ['batting_team', 'bowling_team', 'city',
                    'runs_left', 'balls_left', 'wickets',
                    'total_runs_x', 'crr', 'rrr']
    temp_df = match[feature_cols].copy()

    # 3. Clean infs/NaNs (numeric columns only, avoids pandas downcasting FutureWarning)
    numeric_cols = ['runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']
    for col in numeric_cols:
        # Convert to float first, then replace inf -> NaN -> 0 in one typed pass.
        # Using a per-column numeric assign avoids the deprecated `replace` downcast path.
        col_vals = pd.to_numeric(temp_df[col], errors='coerce').astype(float)
        col_vals = col_vals.replace([np.inf, -np.inf], np.nan).fillna(0)
        temp_df[col] = col_vals
    temp_df['city'] = temp_df['city'].fillna('Unknown')

    # Remove cases where balls_left == 0 to prevent downstream RRR calculations from failing
    temp_df = temp_df[temp_df['balls_left'] != 0]

    if temp_df.empty:
        print(f"No valid match events left to analyze after data cleaning for match_id: {match_id}")
        return None, None

    # 4. Predict over-by-over win/loss probabilities
    try:
        probabilities = model.predict_proba(temp_df)
    except ValueError as e:
        print("⚠️ Model prediction error in match progression analysis:", e)
        return None, None

    # 5. Build progression stats DataFrame
    temp_df['lose'] = np.round(probabilities[:, 0] * 100, 1)
    temp_df['win'] = np.round(probabilities[:, 1] * 100, 1)
    
    # Add over number index
    temp_df['end_of_over'] = np.arange(1, len(temp_df) + 1)

    # 6. Calculate runs scored in each over
    target = temp_df['total_runs_x'].iloc[0]
    runs_left = temp_df['runs_left'].values
    prev_runs = np.insert(runs_left[:-1], 0, target)
    temp_df['runs_after_over'] = prev_runs - runs_left

    # 7. Calculate wickets lost in each over
    wickets = temp_df['wickets'].values
    prev_wickets = np.insert(wickets[:-1], 0, 10)
    temp_df['wickets_in_over'] = prev_wickets - wickets

    # 8. Filter and return clean progression tracking columns
    display_cols = ['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']
    progression_df = temp_df[display_cols].copy()
    
    return progression_df, target

if __name__ == '__main__':
    # Interactive CLI Predictor
    try:
        model = load_trained_model('pipe_lr.pkl')
        print("==================================================")
        print("      IPL WIN PREDICTOR - INTERACTIVE CLI         ")
        print("==================================================\n")
        
        teams = [
            'Sunrisers Hyderabad',
            'Mumbai Indians',
            'Royal Challengers Bangalore',
            'Kolkata Knight Riders',
            'Kings XI Punjab',
            'Chennai Super Kings',
            'Rajasthan Royals',
            'Delhi Capitals'
        ]
        
        print("Select Batting Team:")
        for idx, team in enumerate(teams, 1):
            print(f"{idx}. {team}")
        bat_idx = int(input("\nEnter choice (1-8): "))
        batting_team = teams[bat_idx - 1]
        
        print("\nSelect Bowling Team:")
        remaining_teams = [t for t in teams if t != batting_team]
        for idx, team in enumerate(remaining_teams, 1):
            print(f"{idx}. {team}")
        bowl_idx = int(input("\nEnter choice (1-7): "))
        bowling_team = remaining_teams[bowl_idx - 1]
        
        city = input("\nEnter Host City (e.g. Mumbai, Kolkata, Chennai) [Default: Mumbai]: ").strip()
        if not city:
            city = 'Mumbai'
            
        target = int(input("\nEnter Target Score (Runs to Win): "))
        runs_scored = int(input("Enter Current Runs Scored by Chasing Team: "))
        wickets_lost = int(input("Enter Wickets Fallen (0-9): "))
        
        overs_input = input("Enter Overs Bowled (e.g. 12.3, 15): ")
        if '.' in overs_input:
            parts = overs_input.split('.')
            overs = int(parts[0])
            balls = int(parts[1])
        else:
            overs = int(overs_input)
            balls = 0
            
        balls_faced = (overs * 6) + balls
        balls_left = 120 - balls_faced
        runs_left = target - runs_scored
        wickets_left = 10 - wickets_lost
        
        crr, rrr = calculate_crr_rrr(runs_scored, balls_faced, target)
        
        win_p, lose_p = predict_win_probability(
            model=model,
            batting_team=batting_team,
            bowling_team=bowling_team,
            city=city,
            runs_left=runs_left,
            balls_left=balls_left,
            wickets=wickets_left,
            total_runs_x=target,
            crr=crr,
            rrr=rrr
        )
        
        print("\n==================================================")
        print("              PREDICTION RESULTS                  ")
        print("==================================================")
        print(f"🏏 Chasing Team:  {batting_team}")
        print(f"🏏 Bowling Team:  {bowling_team}")
        print(f"📍 City:          {city}")
        print(f"🎯 Target:        {target} runs")
        print(f"📊 Current Score:  {runs_scored}/{wickets_lost} in {overs}.{balls} overs")
        print(f"📈 CRR:           {crr:.2f} | RRR: {rrr:.2f}")
        print("--------------------------------------------------")
        print(f"🔥 {batting_team} Win Probability: {win_p:.2f}%")
        print(f"🛡️ {bowling_team} Win Probability:   {lose_p:.2f}%")
        print("==================================================\n")
        
    except ValueError as ve:
        print(f"\n⚠️ Input Error: Please enter valid numbers. ({ve})")
    except IndexError:
        print("\n⚠️ Selection Error: Please choose numbers within the specified range.")
    except Exception as e:
        print(f"\n⚠️ An error occurred: {e}")
        print("Please train a model and save it as 'pipe_lr.pkl' first.")
