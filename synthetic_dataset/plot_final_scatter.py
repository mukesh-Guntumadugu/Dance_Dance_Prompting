import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

models = ['DeepResonance', 'MuMu', 'Flamingo', 'Qwen', 'Librosa']

for model in models:
    file_path = f"{model}_full_song_wav_rmse.csv"
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_csv(file_path)

    # Create extremely wide and tall figure so every single tick is visible
    plt.figure(figsize=(60, 40)) 

    color_groups = {
        'Near Perfect (Error <= 0.5) [Dark Green]': {'color': 'darkgreen', 'x': [], 'y': [], 's': 100},
        'Over-predicted (0.5 < Error <= 1.0) [Cyan]': {'color': 'cyan', 'x': [], 'y': [], 's': 80},
        'Over-predicted (1.0 < Error <= 5.0) [Blue]': {'color': 'blue', 'x': [], 'y': [], 's': 80},
        'Under-predicted (0.5 < Error <= 1.0) [Yellow]': {'color': '#cccc00', 'x': [], 'y': [], 's': 80}, 
        'Under-predicted (1.0 < Error <= 5.0) [Brown]': {'color': 'saddlebrown', 'x': [], 'y': [], 's': 80},
        'Massive Error (> 5.0) [Black]': {'color': 'black', 'x': [], 'y': [], 's': 80},
        'Half-BPM Error (Pred * 2 within 1) [Purple]': {'color': 'purple', 'x': [], 'y': [], 's': 90},
        'Half-BPM Error (1 < Pred * 2 error <= 5) [Pink]': {'color': 'hotpink', 'x': [], 'y': [], 's': 90},
        'Double-BPM Error (Pred / 2 within 5) [Orange]': {'color': 'orange', 'x': [], 'y': [], 's': 90}
    }

    for _, row in df.iterrows():
        actual = row['actual_bpm']
        pred = row['pred_bpm']
        error = abs(actual - pred)
        
        # Determine color based on prediction
        if abs(pred * 2 - actual) <= 1.0:
            group_name = 'Half-BPM Error (Pred * 2 within 1) [Purple]'
        elif abs(pred * 2 - actual) <= 5.0:
            group_name = 'Half-BPM Error (1 < Pred * 2 error <= 5) [Pink]'
        elif abs(pred / 2 - actual) <= 5.0:
            group_name = 'Double-BPM Error (Pred / 2 within 5) [Orange]'
        elif error <= 0.5:
            group_name = 'Near Perfect (Error <= 0.5) [Dark Green]'
        elif pred > actual:
            if error <= 1.0:
                group_name = 'Over-predicted (0.5 < Error <= 1.0) [Cyan]'
            elif error <= 5.0:
                group_name = 'Over-predicted (1.0 < Error <= 5.0) [Blue]'
            else:
                group_name = 'Massive Error (> 5.0) [Black]'
        elif pred < actual:
            if error <= 1.0:
                group_name = 'Under-predicted (0.5 < Error <= 1.0) [Yellow]'
            elif error <= 5.0:
                group_name = 'Under-predicted (1.0 < Error <= 5.0) [Brown]'
            else:
                group_name = 'Massive Error (> 5.0) [Black]'
            
        color_groups[group_name]['x'].append(actual)
        color_groups[group_name]['y'].append(error) # Y-AXIS IS NOW THE ERROR

    # Plot each group
    for label, data in color_groups.items():
        if data['x']:  # Only plot if there is data
            plt.scatter(data['x'], data['y'], c=data['color'], s=data['s'], label=label, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Theoretical lines for errors
    plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.5, label='Perfect Prediction (Error = 0)')
    plt.plot([60, 240], [30, 120], color='pink', linestyle='--', alpha=0.5, label='Exact Half-BPM Error (y = 0.5x)')
    plt.plot([60, 120], [60, 120], color='orange', linestyle='--', alpha=0.5, label='Exact Double-BPM Error (y = x)')

    plt.title(f'{model} Individual RMS Error vs. Actual BPM', fontsize=24)
    plt.xlabel('Actual BPM', fontsize=18)
    plt.ylabel('RMS Error (Absolute Difference)', fontsize=18)

    # X-axis formatting
    plt.xticks(np.arange(60, 241, 1), rotation=90, fontsize=12)
    
    # Calculate max error for Y-axis bounds
    max_error = 0
    for data in color_groups.values():
        if data['y']:
            group_max = max(data['y'])
            if group_max > max_error:
                max_error = group_max
    max_y = int(np.ceil(max_error)) + 5
    
    # Y-axis formatting: tick every 1 unit
    plt.yticks(np.arange(0, max_y, 1), fontsize=10)
    
    plt.xlim(59, 241)
    plt.ylim(-1, max_y)

    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(fontsize=16, loc='upper left')

    plt.tight_layout()
    output_name = f'{model}_Final_Scatter.png'
    plt.savefig(output_name, dpi=150)
    print(f"Saved {output_name}")
    plt.close()
