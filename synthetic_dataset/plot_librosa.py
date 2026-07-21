import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv('Librosa_full_song_wav_rmse.csv')

# Group by actual BPM
grouped = df.groupby('actual_bpm')

actual_bpms = []
rmses = []
is_low_ball = []
is_double_ball = []
is_normal = []

for actual_bpm, group in grouped:
    # Compute RMSE for this specific BPM song
    sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
    rmse = np.sqrt(sq_errs.mean())
    
    # Check if the mean prediction was low-balling or double-balling
    mean_pred = group['pred_bpm'].mean()
    
    low_ball = (mean_pred > (actual_bpm * 0.40)) and (mean_pred < (actual_bpm * 0.60))
    double_ball = (mean_pred > (actual_bpm * 1.80)) and (mean_pred < (actual_bpm * 2.20))
    
    actual_bpms.append(actual_bpm)
    rmses.append(rmse)
    is_low_ball.append(low_ball)
    is_double_ball.append(double_ball)
    is_normal.append(not low_ball and not double_ball)

actual_bpms = np.array(actual_bpms)
rmses = np.array(rmses)
is_low_ball = np.array(is_low_ball)
is_double_ball = np.array(is_double_ball)
is_normal = np.array(is_normal)

plt.figure(figsize=(12, 8))

# Plot the theoretical lines for errors
# If it predicts perfectly, error is 0
plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.6, label='Perfect Prediction (RMSE = 0)')

# If it consistently low-balls (pred = 0.5 * actual), RMSE = 0.5 * actual
plt.plot([60, 240], [30, 120], color='pink', linestyle='--', alpha=0.6, label='Half BPM Error (RMSE = 0.5x)')

# If it consistently double-balls (pred = 2.0 * actual), RMSE = actual
plt.plot([60, 120], [60, 120], color='orange', linestyle='--', alpha=0.6, label='Double BPM Error (RMSE = 1.0x)')

# Scatter plot
plt.scatter(actual_bpms[is_normal], rmses[is_normal], color='blue', label='Normal/Other Errors', alpha=0.7)
plt.scatter(actual_bpms[is_low_ball], rmses[is_low_ball], color='hotpink', label='Low-balling (~Half BPM)', alpha=0.9, edgecolor='black', s=80)
plt.scatter(actual_bpms[is_double_ball], rmses[is_double_ball], color='orange', label='Double BPM', alpha=0.9, edgecolor='black', s=80)

plt.title('Librosa: Root Mean Square Error vs BPM (Color-coded by Error Type)')
plt.xlabel('Actual BPM')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.xlim(55, 245)
plt.ylim(-5, 140)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Librosa_RMSE_Colored_plot.png')
print("Plot saved as Librosa_RMSE_Colored_plot.png")
