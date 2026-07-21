import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

models = ['Flamingo', 'Qwen', 'DeepResonance', 'MuMu']
bpms = list(range(60, 241))

fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
axes = axes.flatten()

for idx, m in enumerate(models):
    csv_file = f'{m}_full_song_wav_rmse.csv'
    ax = axes[idx]
    
    if not os.path.exists(csv_file):
        ax.set_title(f'{m} (Data Missing)')
        continue
        
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['actual_bpm', 'pred_bpm'])
    df['error'] = abs(df['pred_bpm'] - df['actual_bpm'])
    
    acc0_list = []
    acc1_list = []
    acc2_list = []
    valid_bpms = []
    
    # Calculate for every single exact BPM
    for bpm in bpms:
        subset = df[df['actual_bpm'] == bpm]
        if len(subset) == 0:
            continue
            
        acc0 = (subset['error'] <= 0.0).sum() / len(subset) * 100
        acc1 = (subset['error'] <= 1.0).sum() / len(subset) * 100
        acc2 = (subset['error'] <= 2.0).sum() / len(subset) * 100
        
        valid_bpms.append(bpm)
        acc0_list.append(acc0)
        acc1_list.append(acc1)
        acc2_list.append(acc2)
        
    # Optional: Smooth the lines slightly using a rolling average of 5 so it's readable
    # because 9 chunks per BPM creates highly jagged 0/11/22/33% jumps
    window = 5
    acc0_smooth = pd.Series(acc0_list).rolling(window, center=True, min_periods=1).mean()
    acc1_smooth = pd.Series(acc1_list).rolling(window, center=True, min_periods=1).mean()
    acc2_smooth = pd.Series(acc2_list).rolling(window, center=True, min_periods=1).mean()
    
    ax.plot(valid_bpms, acc0_smooth, label='Acc_0 (Exact)', color='red', linewidth=2)
    ax.plot(valid_bpms, acc1_smooth, label='Acc_1 (±1 BPM)', color='orange', linewidth=2)
    ax.plot(valid_bpms, acc2_smooth, label='Acc_2 (±2 BPM)', color='green', linewidth=2)
    
    ax.set_title(f'{m} Accuracy Across BPM Spectrum', fontsize=16)
    ax.set_xlim(60, 240)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if idx >= 2:
        ax.set_xlabel('Target BPM (60 to 240)', fontsize=14)
    if idx % 2 == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        
    ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('Acc_Comparison_Detailed.png', dpi=200)
print('Saved Acc_Comparison_Detailed.png')
