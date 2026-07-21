import pandas as pd
import numpy as np
import os
import glob

# Gather all relevant CSVs, or specific ones if needed.
# Using glob to catch all _rmse.csv files that have pred_bpm and actual_bpm
csv_files = glob.glob("*_rmse.csv")

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        continue
        
    if 'actual_bpm' not in df.columns or 'pred_bpm' not in df.columns:
        continue
    
    # Drop the old 'rmse' column that had the averages
    if 'rmse' in df.columns:
        df.drop(columns=['rmse'], inplace=True)
    
    # Calculate the exact error for each individual window (Absolute Error)
    df['rmse'] = abs(df['pred_bpm'] - df['actual_bpm'])
    
    # Accuracy 0: exact match rounded to nearest whole number
    acc0_condition = (np.round(df['pred_bpm']) == np.round(df['actual_bpm']))
    df['acc0'] = acc0_condition.astype(int)
    
    # Accuracy 1: within +/- 4%, BUT NOT an exact match
    def is_within_4_percent(pred, target):
        return abs(pred - target) <= 0.04 * target
        
    acc1_condition = is_within_4_percent(df['pred_bpm'], df['actual_bpm']) & ~acc0_condition
    df['acc1'] = acc1_condition.astype(int)
    
    # Accuracy 2: octave errors within +/- 4%, BUT NOT already counted in Acc 0 or Acc 1
    acc2_condition = (
        is_within_4_percent(df['pred_bpm'], df['actual_bpm'] * 2) |
        is_within_4_percent(df['pred_bpm'], df['actual_bpm'] * 3) |
        is_within_4_percent(df['pred_bpm'], df['actual_bpm'] / 2) |
        is_within_4_percent(df['pred_bpm'], df['actual_bpm'] / 3)
    ) & ~acc1_condition & ~acc0_condition
    df['acc2'] = acc2_condition.astype(int)
    
    # Save back over the original file
    df.to_csv(file_path, index=False)
    
    # Print summary statistics
    acc0_pct = df['acc0'].mean() * 100
    acc1_pct = df['acc1'].mean() * 100
    acc2_pct = df['acc2'].mean() * 100
    print(f"Updated {file_path}")
    print(f"  Accuracy0: {acc0_pct:.2f}%")
    print(f"  Accuracy1: {acc1_pct:.2f}%")
    print(f"  Accuracy2: {acc2_pct:.2f}%\n")
