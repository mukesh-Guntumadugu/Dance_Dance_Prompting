import pandas as pd
import matplotlib.pyplot as plt
import os

models = ['Flamingo', 'Qwen', 'DeepResonance', 'MuMu']
colors = {'Qwen': 'blue', 'Flamingo': 'orange', 'MuMu': 'red', 'DeepResonance': 'green'}
markers = {'Qwen': 'o', 'Flamingo': 's', 'MuMu': '^', 'DeepResonance': 'd'}

ranges = [(60,79), (80,99), (100,119), (120,139), (140,159), (160,179), (180,199), (200,219), (220,240)]
labels = [f'{s}-{e}' for s, e in ranges]

plt.figure(figsize=(10, 6))

for m in models:
    csv_file = f'{m}_full_song_wav_rmse.csv'
    if not os.path.exists(csv_file):
        print(f"Missing {csv_file}")
        continue
    
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['actual_bpm', 'pred_bpm'])
    
    acc2_vals = []
    for (start, end) in ranges:
        subset = df[(df['actual_bpm'] >= start) & (df['actual_bpm'] <= end)].copy()
        if len(subset) == 0:
            acc2_vals.append(0)
            continue
        
        subset['error'] = abs(subset['pred_bpm'] - subset['actual_bpm'])
        acc2 = (subset['error'] <= 2.0).sum() / len(subset) * 100
        acc2_vals.append(acc2)
        
    plt.plot(labels, acc2_vals, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=8)

plt.title('Accuracy (within 2 BPM) Across Different Tempo Ranges', fontsize=16)
plt.xlabel('Target BPM Range', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Accuracy_By_Range.png', dpi=200)
print('Saved Accuracy_By_Range.png')
