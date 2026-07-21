import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

files = glob.glob("*_rmse.csv")
data = []
for f in files:
    try:
        df = pd.read_csv(f)
    except:
        continue
    if 'acc1' not in df.columns:
        continue
    
    model = f.split('_')[0]
    
    if 'stateless_chunk' in f:
        task = 'Stateless Chunk'
    elif 'full_song' in f:
        task = 'Full Song'
    else:
        continue
        
    fmt = 'Unknown'
    if 'wav' in f: fmt = 'WAV'
    elif 'mp3' in f: fmt = 'MP3'
    elif 'ogg' in f: fmt = 'OGG'
    
    if fmt == 'Unknown':
        continue
        
    acc0 = df['acc0'].mean() * 100
    acc1 = df['acc1'].mean() * 100
    acc2 = df['acc2'].mean() * 100
    data.append({'Model': model, 'Task': task, 'Format': fmt, 'Accuracy 0': acc0, 'Accuracy 1': acc1, 'Accuracy 2': acc2})

df_all = pd.DataFrame(data)
summary = df_all.groupby(['Model', 'Task', 'Format'])[['Accuracy 0', 'Accuracy 1', 'Accuracy 2']].mean().reset_index()

# Sort by model to keep it organized
summary = summary.sort_values(by=['Model', 'Task', 'Format'])

# Create a pivot-like label for the Y axis
summary['Condition'] = summary['Model'] + ' - ' + summary['Task'] + ' (' + summary['Format'] + ')'

# Prepare data for heatmap
heatmap_data = summary.set_index('Condition')[['Accuracy 0', 'Accuracy 1', 'Accuracy 2']]

plt.figure(figsize=(10, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
plt.title("Model Accuracies Across Formats and Tasks", pad=20, fontsize=14)
plt.ylabel("")
plt.tight_layout()

# Save directly to the artifacts directory
out_path = "/Users/mukeshguntumadugu/.gemini/antigravity-ide/brain/9cecdee9-f1ad-4a5f-bbfc-09ac3274f22a/accuracy_heatmap.png"
plt.savefig(out_path, dpi=150)
print(f"Heatmap saved to {out_path}")
