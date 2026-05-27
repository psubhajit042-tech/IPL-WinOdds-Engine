# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Visualization Module
Author: SUBHAJIT

This module provides data visualization helpers.
It contains functions to plot classification performance comparisons between models,
and to plot premium, highly interpretable match progression dashboards showing
over-by-over runs scored, wickets fallen, and win/loss probability curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(metrics_df, save_path='model_comparison.png'):
    """
    Plots a multi-bar chart comparing Logistic Regression vs Random Forest on
    Accuracy, Precision, Recall, and F1 Score.
    Optionally saves the plot as a PNG image.
    """
    if metrics_df is None or metrics_df.empty:
        print("No metrics data provided for comparison plotting.")
        return
        
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Standard custom professional colors (viridis palette or customized)
    ax = sns.barplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        data=metrics_df, 
        palette={'Logistic Regression': '#4a90e2', 'Random Forest': '#2ecc71'}
    )
    
    plt.title('Comparison of Model Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Metric', fontsize=12, labelpad=10)
    plt.ylabel('Score', fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=10, fontweight='bold')
                        
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to: {save_path}")
        
    plt.show()

def plot_match_progression(progression_df, target, match_id=None, save_path=None):
    """
    Produces the standard professional IPL Match Win/Lose Probability progression dashboard.
    Plots runs scored (blue bars), wickets fallen (yellow dots/line), and probabilities (green/red lines).
    """
    if progression_df is None or progression_df.empty:
        print("No valid match progression dataframe provided for plotting.")
        return
        
    plt.figure(figsize=(18, 9))
    sns.set_style("darkgrid")
    
    # 1. Bar Chart for runs scored in each over
    plt.bar(
        progression_df['end_of_over'],
        progression_df['runs_after_over'],
        color='#3498db',
        alpha=0.6,
        edgecolor='#2980b9',
        label='Runs Scored per Over'
    )
    
    # 2. Line Chart for wickets fallen in each over (yellow dots/line)
    # We multiply by a constant or put on secondary axis, but standard notebook matches runs:
    plt.plot(
        progression_df['end_of_over'],
        progression_df['wickets_in_over'],
        color='#f1c40f',
        linewidth=3,
        marker='o',
        markersize=8,
        label='Wickets in Over'
    )
    
    # 3. Win Probability Curve (Green line)
    plt.plot(
        progression_df['end_of_over'],
        progression_df['win'],
        color='#2ecc71',
        linewidth=4.5,
        label='Win Probability (%)'
    )
    
    # 4. Lose Probability Curve (Red line)
    plt.plot(
        progression_df['end_of_over'],
        progression_df['lose'],
        color='#e74c3c',
        linewidth=4.5,
        label='Lose Probability (%)'
    )
    
    # Title & Labels
    match_title = f"Match Progression Analysis (Target: {target})"
    if match_id:
        match_title = f"IPL Match {match_id} Progression (Target: {target})"
        
    plt.title(match_title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('End of Over', fontsize=14, labelpad=10)
    plt.ylabel('Percentage (%) / Counts', fontsize=14, labelpad=10)
    
    plt.xticks(progression_df['end_of_over'], fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Legend
    plt.legend(fontsize=12, loc='upper left', frameon=True, facecolor='white', framealpha=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved match progression plot to: {save_path}")
        
    plt.show()

if __name__ == '__main__':
    # Local execution demo with synthetic data
    import pandas as pd
    print("Testing visualization modules with synthetic data...")
    
    # Mock metrics
    mock_metrics = pd.DataFrame([
        {'Model': 'Logistic Regression', 'Metric': 'Accuracy', 'Score': 0.8145},
        {'Model': 'Logistic Regression', 'Metric': 'Precision', 'Score': 0.8112},
        {'Model': 'Logistic Regression', 'Metric': 'Recall', 'Score': 0.8240},
        {'Model': 'Logistic Regression', 'Metric': 'F1 Score', 'Score': 0.8175},
        {'Model': 'Random Forest', 'Metric': 'Accuracy', 'Score': 0.9986},
        {'Model': 'Random Forest', 'Metric': 'Precision', 'Score': 0.9982},
        {'Model': 'Random Forest', 'Metric': 'Recall', 'Score': 0.9990},
        {'Model': 'Random Forest', 'Metric': 'F1 Score', 'Score': 0.9986}
    ])
    
    # Mock progression
    mock_progression = pd.DataFrame({
        'end_of_over': list(range(1, 11)),
        'runs_after_over': [6, 12, 5, 8, 15, 7, 11, 4, 18, 9],
        'wickets_in_over': [0, 1, 0, 0, 2, 0, 0, 1, 0, 1],
        'lose': [60, 65, 72, 68, 85, 80, 72, 83, 40, 52],
        'win': [40, 35, 28, 32, 15, 20, 28, 17, 60, 48]
    })
    
    try:
        plot_model_comparison(mock_metrics, save_path='mock_comparison.png')
        plot_match_progression(mock_progression, target=160, match_id=999, save_path='mock_progression.png')
        print("Visualization tests completed successfully! Mock images generated.")
    except Exception as e:
        print(f"Error drawing mock plots: {e}")
