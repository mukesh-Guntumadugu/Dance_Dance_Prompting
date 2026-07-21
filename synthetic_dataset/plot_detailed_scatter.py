import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the first run
df = pd.read_csv('Librosa_full_song_wav_run1_rmse.csv')

# Group by actual BPM to calculate RMSE and get the mean predicted BPM
grouped = df.groupby('actual_bpm')

# Create extremely wide and tall figure so every single tick is visible
plt.figure(figsize=(60, 30)) 

color_groups = {
    'Near Perfect (Error <= 0.5) [Dark Green]': {'color': 'darkgreen', 'x': [], 'y': [], 's': 100},
    'Over-predicted (0.5 < Error <= 1.0) [Cyan]': {'color': 'cyan', 'x': [], 'y': [], 's': 80},
    'Over-predicted (1.0 < Error <= 5.0) [Blue]': {'color': 'blue', 'x': [], 'y': [], 's': 80},
    'Over-predicted (Error > 5.0) [Navy]': {'color': 'navy', 'x': [], 'y': [], 's': 80},
    'Under-predicted (0.5 < Error <= 1.0) [Yellow]': {'color': '#cccc00', 'x': [], 'y': [], 's': 80}, # Using a darker yellow for visibility
    'Under-predicted (1.0 < Error <= 5.0) [Brown]': {'color': 'saddlebrown', 'x': [], 'y': [], 's': 80},
    'Under-predicted (Error > 5.0) [Red]': {'color': 'red', 'x': [], 'y': [], 's': 80},
    'Half-BPM Error (Pred * 2 within 1) [Purple]': {'color': 'purple', 'x': [], 'y': [], 's': 90},
    'Half-BPM Error (1 < Pred * 2 error <= 5) [Pink]': {'color': 'hotpink', 'x': [], 'y': [], 's': 90},
    'Double-BPM Error (Pred / 2 within 5) [Orange]': {'color': 'orange', 'x': [], 'y': [], 's': 90}
}

for actual, group in grouped:
    # Calculate RMSE for the Y-axis
    sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
    rmse = np.sqrt(sq_errs.mean())
    
    # Get mean predicted BPM to determine the color
    pred = group['pred_bpm'].mean()
    error = abs(actual - pred)
    
    # Determine color
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
            group_name = 'Over-predicted (Error > 5.0) [Navy]'
    elif pred < actual:
        if error <= 1.0:
            group_name = 'Under-predicted (0.5 < Error <= 1.0) [Yellow]'
        elif error <= 5.0:
            group_name = 'Under-predicted (1.0 < Error <= 5.0) [Brown]'
        else:
            group_name = 'Under-predicted (Error > 5.0) [Red]'
        
    color_groups[group_name]['x'].append(actual)
    color_groups[group_name]['y'].append(rmse)

# Plot each group
for label, data in color_groups.items():
    if data['x']:  # Only plot if there is data
        plt.scatter(data['x'], data['y'], c=data['color'], s=data['s'], label=label, alpha=0.8, edgecolors='black', linewidth=0.5)

# Theoretical lines
plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.5, label='Perfect Prediction (RMSE = 0)')
plt.plot([60, 240], [30, 120], color='pink', linestyle='--', alpha=0.5, label='Expected RMSE if Half-BPM Error (RMSE = 0.5x)')

plt.title('Librosa RMSE vs. Actual BPM (Over/Under Prediction Split)', fontsize=24)
plt.xlabel('Actual BPM', fontsize=18)
plt.ylabel('Root Mean Square Error (RMSE)', fontsize=18)

# The user wants EVERY SINGLE BPM on the x-axis, no matter how wide the graph is
plt.xticks(np.arange(60, 241, 1), rotation=90, fontsize=12)
# And EVERY SINGLE RMSE value on the y-axis
plt.yticks(np.arange(0, 141, 1), fontsize=10)

plt.xlim(59, 241)
plt.ylim(-5, 140)

plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(fontsize=16, loc='upper left')

plt.tight_layout()
plt.savefig('Librosa_Detailed_RMSE.png', dpi=150)
print("Saved Librosa_Detailed_RMSE.png")
