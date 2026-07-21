import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

formats = ['wav', 'mp3', 'ogg']
runs = range(1, 11)

# Seed for reproducibility of the visual jitter
np.random.seed(42)

for ext in formats:
    print(f"Generating jittered pancaked graph for {ext.upper()}...")
    plt.figure(figsize=(60, 30)) 
    
    color_groups = {
        'Near Perfect (Error <= 0.5) [Dark Green]': {'color': 'darkgreen', 'x': [], 'y': [], 's': 100},
        'Over-predicted (0.5 < Error <= 1.0) [Cyan]': {'color': 'cyan', 'x': [], 'y': [], 's': 80},
        'Over-predicted (1.0 < Error <= 5.0) [Blue]': {'color': 'blue', 'x': [], 'y': [], 's': 80},
        'Over-predicted (Error > 5.0) [Navy]': {'color': 'navy', 'x': [], 'y': [], 's': 80},
        'Under-predicted (0.5 < Error <= 1.0) [Yellow]': {'color': '#cccc00', 'x': [], 'y': [], 's': 80}, 
        'Under-predicted (1.0 < Error <= 5.0) [Brown]': {'color': 'saddlebrown', 'x': [], 'y': [], 's': 80},
        'Under-predicted (Error > 5.0) [Red]': {'color': 'red', 'x': [], 'y': [], 's': 80},
        'Half-BPM Over-predicted (Pred > True Half, within 1) [Light Purple]': {'color': '#dca3ff', 'x': [], 'y': [], 's': 90},
        'Half-BPM Under-predicted (Pred < True Half, within 1) [Dark Purple]': {'color': '#4a0080', 'x': [], 'y': [], 's': 90},
        'Half-BPM Over-predicted (Pred > True Half, 1-5 error) [Light Pink]': {'color': '#ffb3cc', 'x': [], 'y': [], 's': 90},
        'Half-BPM Under-predicted (Pred < True Half, 1-5 error) [Deep Pink]': {'color': '#cc0052', 'x': [], 'y': [], 's': 90},
        'Double-BPM Over-predicted (Pred > True Double) [Light Orange]': {'color': '#ffb366', 'x': [], 'y': [], 's': 90},
        'Double-BPM Under-predicted (Pred < True Double) [Dark Orange]': {'color': '#cc5200', 'x': [], 'y': [], 's': 90}
    }
    
    for run in runs:
        csv_file = f'Librosa_full_song_{ext}_run{run}_rmse.csv'
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        grouped = df.groupby('actual_bpm')
        
        for actual, group in grouped:
            sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
            rmse = np.sqrt(sq_errs.mean())
            
            pred = group['pred_bpm'].mean()
            error = abs(actual - pred)
            
            true_half = actual / 2.0
            true_double = actual * 2.0
            
            if abs(pred - true_half) <= 0.5:
                if pred >= true_half:
                    group_name = 'Half-BPM Over-predicted (Pred > True Half, within 1) [Light Purple]'
                else:
                    group_name = 'Half-BPM Under-predicted (Pred < True Half, within 1) [Dark Purple]'
            elif abs(pred - true_half) <= 2.5:
                if pred >= true_half:
                    group_name = 'Half-BPM Over-predicted (Pred > True Half, 1-5 error) [Light Pink]'
                else:
                    group_name = 'Half-BPM Under-predicted (Pred < True Half, 1-5 error) [Deep Pink]'
            elif abs(pred - true_double) <= 5.0:
                if pred >= true_double:
                    group_name = 'Double-BPM Over-predicted (Pred > True Double) [Light Orange]'
                else:
                    group_name = 'Double-BPM Under-predicted (Pred < True Double) [Dark Orange]'
            elif error <= 0.5:
                group_name = 'Near Perfect (Error <= 0.5) [Dark Green]'
            elif pred > actual:
                if error <= 1.0:
                    group_name = 'Over-predicted (0.5 < Error <= 1.0) [Cyan]'
                elif error <= 5.0:
                    group_name = 'Over-predicted (1.0 < Error <= 5.0) [Blue]'
                else:
                    group_name = 'Over-predicted (Error > 5.0) [Navy]'
            elif pred < actual:
                if error <= 1.0:
                    group_name = 'Under-predicted (0.5 < Error <= 1.0) [Yellow]'
                elif error <= 5.0:
                    group_name = 'Under-predicted (1.0 < Error <= 5.0) [Brown]'
                else:
                    group_name = 'Under-predicted (Error > 5.0) [Red]'
            else:
                group_name = 'Under-predicted (Error > 5.0) [Red]' 
                
            # ADD JITTER HERE SO THE 10 POINTS DON'T PERFECTLY OVERLAP
            jitter_x = actual + np.random.uniform(-0.15, 0.15)
            jitter_y = rmse + np.random.uniform(-0.15, 0.15)
            
            color_groups[group_name]['x'].append(jitter_x)
            color_groups[group_name]['y'].append(jitter_y)
            
    # Plot each group
    for label, data in color_groups.items():
        if data['x']:
            plt.scatter(data['x'], data['y'], c=data['color'], s=50, label=label, alpha=0.5, edgecolors='black', linewidth=0.5)
    
    plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.5, label='Perfect Prediction (RMSE = 0)')
    plt.plot([60, 240], [30, 120], color='pink', linestyle='--', alpha=0.5, label='Expected RMSE if Half-BPM Error (RMSE = 0.5x)')
    
    plt.title(f'Librosa {ext.upper()} All 10 Runs Pancaked (WITH JITTER) - RMSE vs. Actual BPM', fontsize=24)
    plt.xlabel('Actual BPM', fontsize=18)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=18)
    
    plt.xticks(np.arange(60, 241, 1), rotation=90, fontsize=12)
    plt.yticks(np.arange(0, 141, 1), fontsize=10)
    
    plt.xlim(59, 241)
    plt.ylim(-5, 140)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(fontsize=16, loc='upper left')
    
    plt.tight_layout()
    out_name = f'Librosa_Jittered_{ext.upper()}.png'
    plt.savefig(out_name, dpi=150)
    plt.close()
    print(f"Saved {out_name}")
