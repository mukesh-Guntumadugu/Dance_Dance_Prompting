import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

models = ['DeepResonance', 'MuMu', 'Flamingo', 'Qwen', 'Librosa']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for model, color in zip(models, colors):
    file_path = f"{model}_full_song_wav_rmse.csv"
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    # Calculate squared error
    df['squared_error'] = (df['pred_bpm'] - df['actual_bpm'])**2
    
    # Group by actual_bpm to calculate RMSE for each BPM
    rmse_per_bpm = df.groupby('actual_bpm')['squared_error'].mean().apply(np.sqrt).reset_index()
    rmse_per_bpm.rename(columns={'squared_error': 'rmse'}, inplace=True)
    
    max_rmse = rmse_per_bpm['rmse'].max()
    
    plt.figure(figsize=(30, 15))
    plt.plot(rmse_per_bpm['actual_bpm'], rmse_per_bpm['rmse'], label=model, color=color, alpha=0.8, linewidth=2, marker='o', markersize=3)

    plt.xlabel('BPM (Actual)')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs BPM (60-240) - {model}')
    plt.xticks(np.arange(60, 241, 1), rotation=90, fontsize=8)

    # Set Y-axis to vary by 1
    plt.yticks(np.arange(0, int(np.ceil(max_rmse)) + 2, 1), fontsize=8)

    plt.xlim(59, 241)
    plt.ylim(0, int(np.ceil(max_rmse)) + 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_filename = f"{model}_sweep_rmse.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as {output_filename}")
    plt.close()
