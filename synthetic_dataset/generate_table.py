import pandas as pd
import glob
import re

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
        task = 'Other'
        
    fmt = 'Unknown'
    if 'wav' in f: fmt = 'WAV'
    elif 'mp3' in f: fmt = 'MP3'
    elif 'ogg' in f: fmt = 'OGG'
        
    acc0 = df['acc0'].mean() * 100
    acc1 = df['acc1'].mean() * 100
    acc2 = df['acc2'].mean() * 100
    data.append({'Model': model, 'Task': task, 'Format': fmt, 'Acc0': acc0, 'Acc1': acc1, 'Acc2': acc2})

df_all = pd.DataFrame(data)

summary = df_all.groupby(['Model', 'Task', 'Format'])[['Acc0', 'Acc1', 'Acc2']].mean().reset_index()

print("| Model | Task | Format | Accuracy 0 | Accuracy 1 | Accuracy 2 |")
print("|---|---|---|---|---|---|")
for _, row in summary.sort_values(by=['Model', 'Task', 'Format']).iterrows():
    print(f"| **{row['Model']}** | {row['Task']} | {row['Format']} | {row['Acc0']:.2f}% | {row['Acc1']:.2f}% | {row['Acc2']:.2f}% |")
