import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

formats = ['wav', 'mp3', 'ogg']
runs = range(1, 11)

for ext in formats:
    print(f"Generating 3D graph for {ext.upper()}...")
    
    # We will use a reasonably large figure
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    color_groups = {
        'Near Perfect (Error <= 0.5) [Dark Green]': {'color': 'darkgreen', 'x': [], 'y': [], 'z': [], 's': 80},
        'Over-predicted (0.5 < Error <= 1.0) [Cyan]': {'color': 'cyan', 'x': [], 'y': [], 'z': [], 's': 60},
        'Over-predicted (1.0 < Error <= 5.0) [Blue]': {'color': 'blue', 'x': [], 'y': [], 'z': [], 's': 60},
        'Over-predicted (Error > 5.0) [Navy]': {'color': 'navy', 'x': [], 'y': [], 'z': [], 's': 60},
        'Under-predicted (0.5 < Error <= 1.0) [Yellow]': {'color': '#cccc00', 'x': [], 'y': [], 'z': [], 's': 60}, 
        'Under-predicted (1.0 < Error <= 5.0) [Brown]': {'color': 'saddlebrown', 'x': [], 'y': [], 'z': [], 's': 60},
        'Under-predicted (Error > 5.0) [Red]': {'color': 'red', 'x': [], 'y': [], 'z': [], 's': 60},
        'Half-BPM Over-predicted (within 1) [Light Purple]': {'color': '#dca3ff', 'x': [], 'y': [], 'z': [], 's': 70},
        'Half-BPM Under-predicted (within 1) [Dark Purple]': {'color': '#4a0080', 'x': [], 'y': [], 'z': [], 's': 70},
        'Half-BPM Over-predicted (1-5 error) [Light Pink]': {'color': '#ffb3cc', 'x': [], 'y': [], 'z': [], 's': 70},
        'Half-BPM Under-predicted (1-5 error) [Deep Pink]': {'color': '#cc0052', 'x': [], 'y': [], 'z': [], 's': 70},
        'Double-BPM Over-predicted [Light Orange]': {'color': '#ffb366', 'x': [], 'y': [], 'z': [], 's': 70},
        'Double-BPM Under-predicted [Dark Orange]': {'color': '#cc5200', 'x': [], 'y': [], 'z': [], 's': 70}
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
                    group_name = 'Half-BPM Over-predicted (within 1) [Light Purple]'
                else:
                    group_name = 'Half-BPM Under-predicted (within 1) [Dark Purple]'
            elif abs(pred - true_half) <= 2.5:
                if pred >= true_half:
                    group_name = 'Half-BPM Over-predicted (1-5 error) [Light Pink]'
                else:
                    group_name = 'Half-BPM Under-predicted (1-5 error) [Deep Pink]'
            elif abs(pred - true_double) <= 5.0:
                if pred >= true_double:
                    group_name = 'Double-BPM Over-predicted [Light Orange]'
                else:
                    group_name = 'Double-BPM Under-predicted [Dark Orange]'
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
                
            color_groups[group_name]['x'].append(actual)
            color_groups[group_name]['y'].append(run) # Z-axis is run, Y-axis in plot
            color_groups[group_name]['z'].append(rmse) # Z-axis in plot
            
    # Plot each group
    for label, data in color_groups.items():
        if data['x']:
            ax.scatter(data['x'], data['y'], data['z'], c=data['color'], s=data['s'], label=label, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add theoretical planes or lines if needed, but in 3D it might get messy. Let's skip the expected lines to keep it clean.
    
    ax.set_title(f'Librosa {ext.upper()} 3D RMSE Scatter Plot (Runs 1-10)', fontsize=26, pad=30)
    ax.set_xlabel('Actual BPM', fontsize=20, labelpad=20)
    ax.set_ylabel('Run Number', fontsize=20, labelpad=20)
    ax.set_zlabel('Root Mean Square Error (RMSE)', fontsize=20, labelpad=20)
    
    # Set ticks
    ax.set_xticks(np.arange(60, 241, 10))
    ax.set_yticks(np.arange(1, 11, 1))
    
    ax.set_xlim(59, 241)
    ax.set_ylim(0.5, 10.5)
    ax.set_zlim(-5, 140)
    
    # Adjust viewing angle for best 3D perspective
    ax.view_init(elev=20, azim=-60)
    
    # Move legend outside the plot
    ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    out_name = f'Librosa_3D_{ext.upper()}.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_name}")
