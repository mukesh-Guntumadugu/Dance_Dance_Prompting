import pandas as pd
import matplotlib.pyplot as plt
import os

models = ['Flamingo', 'Qwen', 'DeepResonance', 'MuMu']
colors = {'Qwen': 'blue', 'Flamingo': 'orange', 'MuMu': 'red', 'DeepResonance': 'green'}

bpms = list(range(60, 241))

plt.figure(figsize=(12, 6))

for m in models:
    csv_file = f'{m}_full_song_wav_rmse.csv'
    if not os.path.exists(csv_file):
        continue
        
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['actual_bpm', 'pred_bpm'])
    
    # Calculate absolute error
    df['error'] = abs(df['pred_bpm'] - df['actual_bpm'])
    
    # Get the average error at each exact BPM
    avg_error_per_bpm = []
    valid_bpms = []
    
    for bpm in bpms:
        subset = df[df['actual_bpm'] == bpm]
        if len(subset) == 0:
            continue
        valid_bpms.append(bpm)
        avg_error_per_bpm.append(subset['error'].mean())
        
    # Smooth the lines slightly so it's readable
    smoothed_error = pd.Series(avg_error_per_bpm).rolling(window=5, center=True, min_periods=1).mean()
    
    plt.plot(valid_bpms, smoothed_error, label=m, color=colors[m], linewidth=2)

# Plot the "perfect" line at Y=0
plt.plot([60, 240], [0, 0], color='black', linestyle='--', linewidth=2, label='Perfect Prediction (0 Error)')

plt.title('Average Error vs Target BPM (60 to 240)', fontsize=16)
plt.xlabel('Target BPM', fontsize=14)
plt.ylabel('Average Absolute Error (BPM)', fontsize=14)
plt.xlim(60, 240)
plt.ylim(0, 150) # Assuming max error might be around 150
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('Error_vs_BPM.png', dpi=200)
print('Saved Error_vs_BPM.png')
