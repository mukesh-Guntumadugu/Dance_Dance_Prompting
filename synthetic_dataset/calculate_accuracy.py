import pandas as pd

df = pd.read_csv('Librosa_full_song_wav_run1_rmse.csv')

acc1_count = 0
acc2_count = 0
total = len(df)

for _, row in df.iterrows():
    actual = row['actual_bpm']
    pred = row['pred_bpm']
    
    # Accuracy 1: within 4%
    if abs(pred - actual) <= 0.04 * actual:
        acc1_count += 1
        acc2_count += 1
    # Accuracy 2: within 4% or octave error (x2, x0.5, x3, x0.33)
    else:
        for factor in [2.0, 0.5, 3.0, 1.0/3.0]:
            if abs(pred - (actual * factor)) <= 0.04 * (actual * factor):
                acc2_count += 1
                break

print(f"Accuracy 1 (Exact match within 4%): {acc1_count}/{total} = {(acc1_count/total)*100:.2f}%")
print(f"Accuracy 2 (Includes Octave/Half-BPM errors): {acc2_count}/{total} = {(acc2_count/total)*100:.2f}%")
