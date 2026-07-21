import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv('Librosa_full_song_wav_rmse.csv')

# Group by actual BPM
grouped = df.groupby('actual_bpm')

actual_bpms = []
original_rmses = []
corrected_rmses = []

for actual_bpm, group in grouped:
    # 1. Original Error
    sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
    orig_rmse = np.sqrt(sq_errs.mean())
    
    # 2. Corrected Error
    # We will automatically double the prediction if it's a "low-ball" 
    # and halve it if it's a "double-ball"
    corrected_preds = []
    for p in group['pred_bpm']:
        if (p > (actual_bpm * 0.40)) and (p < (actual_bpm * 0.60)):
            corrected_preds.append(p * 2.0) # Fix the halving error by doubling it!
        elif (p > (actual_bpm * 1.80)) and (p < (actual_bpm * 2.20)):
            corrected_preds.append(p / 2.0) # Fix the doubling error by halving it!
        else:
            corrected_preds.append(p)
            
    corrected_preds = np.array(corrected_preds)
    corr_sq_errs = (corrected_preds - actual_bpm) ** 2
    corr_rmse = np.sqrt(corr_sq_errs.mean())
    
    actual_bpms.append(actual_bpm)
    original_rmses.append(orig_rmse)
    corrected_rmses.append(corr_rmse)

actual_bpms = np.array(actual_bpms)
original_rmses = np.array(original_rmses)
corrected_rmses = np.array(corrected_rmses)

plt.figure(figsize=(12, 8))

# Plot the theoretical lines
plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.6, label='Perfect Prediction (RMSE = 0)')

# Scatter plot of Original vs Corrected
plt.scatter(actual_bpms, original_rmses, color='hotpink', label='Original RMSE (With Octave Errors)', alpha=0.5, s=60)
plt.scatter(actual_bpms, corrected_rmses, color='green', label='Corrected RMSE (Fixed Halving/Doubling)', alpha=0.9, edgecolor='black', s=80)

# Draw lines connecting the original to the corrected to show the drop in error
for i in range(len(actual_bpms)):
    if abs(original_rmses[i] - corrected_rmses[i]) > 5: # Only draw lines for points that were corrected
        plt.plot([actual_bpms[i], actual_bpms[i]], [original_rmses[i], corrected_rmses[i]], color='gray', alpha=0.3)

plt.title('Librosa: Original RMSE vs. Corrected RMSE (Octave Errors Fixed)')
plt.xlabel('Actual BPM')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.xlim(55, 245)
plt.ylim(-5, 140)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Librosa_RMSE_Fixed_plot.png')
print("Plot saved as Librosa_RMSE_Fixed_plot.png")
