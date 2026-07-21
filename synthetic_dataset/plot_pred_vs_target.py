import pandas as pd
import matplotlib.pyplot as plt
import os

models = ['Flamingo', 'Qwen', 'DeepResonance', 'MuMu']
colors = {'Qwen': 'blue', 'Flamingo': 'orange', 'MuMu': 'red', 'DeepResonance': 'green'}

bpms = list(range(60, 241))

plt.figure(figsize=(12, 8))

for m in models:
    csv_file = f'{m}_full_song_wav_rmse.csv'
    if not os.path.exists(csv_file):
        continue
        
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['actual_bpm', 'pred_bpm'])
    
    # Get the average predicted BPM for each exact Target BPM
    avg_pred_per_bpm = []
    valid_bpms = []
    
    for bpm in bpms:
        subset = df[df['actual_bpm'] == bpm]
        if len(subset) == 0:
            continue
        valid_bpms.append(bpm)
        avg_pred_per_bpm.append(subset['pred_bpm'].mean())
        
    # Smooth the lines slightly so it's readable
    smoothed_pred = pd.Series(avg_pred_per_bpm).rolling(window=5, center=True, min_periods=1).mean()
    
    plt.plot(valid_bpms, smoothed_pred, label=m, color=colors[m], linewidth=2.5, alpha=0.8)

# Plot the "perfect" diagonal line (y = x)
plt.plot([60, 240], [60, 240], color='black', linestyle='--', linewidth=3, label='Perfect Prediction (y = x)')

plt.title('Predicted BPM vs Target BPM (60 to 240)', fontsize=18)
plt.xlabel('Target BPM', fontsize=16)
plt.ylabel('Predicted BPM', fontsize=16)

# Make it a perfectly square grid if possible
plt.xlim(60, 240)
plt.ylim(60, 240)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig('Pred_vs_Target_BPM.png', dpi=200)
print('Saved Pred_vs_Target_BPM.png')
