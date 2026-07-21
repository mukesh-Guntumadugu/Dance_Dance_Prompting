import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
FORMATS = ['wav', 'mp3', 'ogg']
ITERATIONS = 10
BPM_RANGE = (55, 245)

def plot_individual_run(fmt, run_idx, actual_bpms, rmses):
    plt.figure(figsize=(12, 8))
    
    # Plot the theoretical lines
    plt.plot([60, 240], [0, 0], color='gray', linestyle='--', alpha=0.6, label='Perfect Prediction (RMSE = 0)')
    plt.plot([60, 240], [30, 120], color='pink', linestyle='--', alpha=0.6, label='Half BPM Error (RMSE = 0.5x)')
    plt.plot([60, 120], [60, 120], color='orange', linestyle='--', alpha=0.6, label='Double BPM Error (RMSE = 1.0x)')
    
    # Scatter plot for this specific run
    plt.scatter(actual_bpms, rmses, color='blue', label=f'RMSE (Run {run_idx})', alpha=0.7, s=50)
    
    plt.title(f'Librosa RMSE vs BPM - Format: {fmt.upper()} - Run {run_idx}')
    plt.xlabel('Actual BPM')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.xlim(BPM_RANGE)
    plt.ylim(-5, 140)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_name = f'Librosa_Stability_{fmt.upper()}_run{run_idx}.png'
    plt.savefig(out_name)
    plt.close()
    print(f"Saved {out_name}")

def generate_individual_plots():
    for fmt in FORMATS:
        for i in range(1, ITERATIONS + 1):
            filename = f'Librosa_full_song_{fmt}_run{i}_rmse.csv'
            if not os.path.exists(filename):
                continue
                
            df = pd.read_csv(filename)
            grouped = df.groupby('actual_bpm')
            
            actual_bpms = []
            rmses = []
            for actual_bpm, group in grouped:
                sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
                rmse = np.sqrt(sq_errs.mean())
                actual_bpms.append(actual_bpm)
                rmses.append(rmse)
                
            plot_individual_run(fmt, i, actual_bpms, rmses)

if __name__ == "__main__":
    generate_individual_plots()
