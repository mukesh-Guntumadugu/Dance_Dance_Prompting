import pandas as pd
import numpy as np
import os

models = ['DeepResonance', 'MuMu', 'Flamingo', 'Qwen', 'Librosa']

for model in models:
    file_path = f"{model}_full_song_wav_rmse.csv"
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    
    # Calculate squared error for each row
    df['squared_error'] = (df['pred_bpm'] - df['actual_bpm'])**2
    
    # Group by actual_bpm to calculate RMSE
    rmse_per_bpm = df.groupby('actual_bpm')['squared_error'].mean().apply(np.sqrt).reset_index()
    rmse_per_bpm.rename(columns={'squared_error': 'rmse'}, inplace=True)
    
    # Save the aggregated RMSE values to a new CSV file
    output_path = f"{model}_aggregated_rmse.csv"
    rmse_per_bpm.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
