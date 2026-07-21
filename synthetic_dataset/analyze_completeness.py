import pandas as pd
import glob

files = glob.glob("*_rmse.csv")

universal_set = set()
file_stats = []

for f in files:
    try:
        df = pd.read_csv(f)
    except:
        continue
    
    if not all(c in df.columns for c in ['song_name', 'window_start', 'window_end']):
        continue
        
    duplicates = df.duplicated(subset=['song_name', 'window_start', 'window_end']).sum()
    
    windows = set(zip(df['song_name'], df['window_start'], df['window_end']))
    universal_set.update(windows)
    
    file_stats.append({
        'File': f,
        'Total Rows': len(df),
        'Unique Windows': len(windows),
        'Duplicates': duplicates,
        'Windows': windows
    })

max_expected = len(universal_set)

print(f"### Overall Expected Windows: {max_expected}\\n")

print("| File | Total Rows | Unique Windows | Missing Windows | Duplicates | Status |")
print("|---|---|---|---|---|---|")

for stat in sorted(file_stats, key=lambda x: x['File']):
    missing = max_expected - stat['Unique Windows']
    status = "✅ Complete"
    if missing > 0 or stat['Duplicates'] > 0:
        status = "❌ Issues Found"
    print(f"| {stat['File']} | {stat['Total Rows']} | {stat['Unique Windows']} | {missing} | {stat['Duplicates']} | {status} |")
