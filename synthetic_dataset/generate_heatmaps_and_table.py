import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_files = glob.glob("*_rmse.csv")
    
    all_data = []
    
    for f in csv_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        model = parts[0]
        
        # Determine format
        fmt = "unknown"
        if "mp3" in basename:
            fmt = "mp3"
        elif "wav" in basename:
            fmt = "wav"
        elif "ogg" in basename:
            fmt = "ogg"
        else:
            # Maybe inside parts
            for p in parts:
                if p in ["mp3", "wav", "ogg"]:
                    fmt = p
                    break
        
        if fmt == "unknown":
            continue
            
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in ['actual_bpm', 'pred_bpm']):
                continue
                
            for _, row in df.iterrows():
                all_data.append({
                    "Model": model,
                    "Format": fmt.upper(),
                    "Actual": row['actual_bpm'],
                    "Pred": row['pred_bpm']
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_data:
        print("No valid data found.")
        return
        
    df_all = pd.DataFrame(all_data)
    
    # 1. Generate Heatmaps per model (aggregating all formats)
    models = df_all['Model'].unique()
    for model in models:
        plt.figure(figsize=(10, 8))
        model_df = df_all[df_all['Model'] == model]
        
        # We can use a 2D histogram / heatmap
        # Define bins from 60 to 240
        bins = np.arange(50, 260, 10)
        h, xedges, yedges, image = plt.hist2d(model_df['Actual'], model_df['Pred'], bins=bins, cmap='viridis', cmin=1)
        plt.colorbar(label='Count')
        
        # Plot y=x line
        plt.plot([60, 240], [60, 240], color='red', linestyle='--', label='Perfect Prediction')
        
        plt.title(f"{model} - BPM Prediction Heatmap")
        plt.xlabel("Actual BPM")
        plt.ylabel("Predicted BPM")
        plt.legend()
        plt.tight_layout()
        
        heatmap_path = f"{model}_overall_heatmap.png"
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Generated heatmap for {model} at {heatmap_path}")
        
    # 2. Compute Table: Accuracy 0, Accuracy 1, Accuracy 2
    # Assuming Acc0 = |diff| <= 0, Acc1 = |diff| <= 1, Acc2 = |diff| <= 2
    df_all['Diff'] = (df_all['Actual'] - df_all['Pred']).abs()
    
    results = []
    
    for model in sorted(models):
        for fmt in ["MP3", "WAV", "OGG"]:
            sub_df = df_all[(df_all['Model'] == model) & (df_all['Format'] == fmt)]
            if len(sub_df) == 0:
                continue
                
            total = len(sub_df)
            acc0 = len(sub_df[sub_df['Diff'] <= 0.5]) / total * 100  # Allowing 0.5 for rounding
            
            # Accuracy 1: ±4 BPM
            acc1 = len(sub_df[sub_df['Diff'] <= 4.0]) / total * 100
            
            # Accuracy 2: Octave errors & ±4 BPM
            def is_acc2(row):
                actual = row['Actual']
                pred = row['Pred']
                if abs(actual - pred) <= 4.0:
                    return True
                if abs(actual * 2 - pred) <= 4.0:
                    return True
                if abs(actual / 2 - pred) <= 4.0:
                    return True
                if abs(actual * 3 - pred) <= 4.0:
                    return True
                if abs(actual / 3 - pred) <= 4.0:
                    return True
                return False
                
            acc2 = sub_df.apply(is_acc2, axis=1).sum() / total * 100
            
            results.append({
                "Model": model,
                "Format": fmt,
                "Acc0_pct": acc0,
                "Acc1_pct": acc1,
                "Acc2_pct": acc2,
                "Total_Samples": total
            })
            
    # Save table to Markdown
    md_lines = ["# BPM Prediction Average Accuracy Table\n"]
    md_lines.append("| Model | Format | Accuracy 0 (±0 BPM) | Accuracy 1 (±4 BPM) | Accuracy 2 (Octave ±4) | Total Samples |")
    md_lines.append("|---|---|---|---|---|---|")
    
    for r in results:
        md_lines.append(f"| {r['Model']} | {r['Format']} | {r['Acc0_pct']:.2f}% | {r['Acc1_pct']:.2f}% | {r['Acc2_pct']:.2f}% | {r['Total_Samples']} |")
        
    with open("accuracy_table.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print("Generated accuracy_table.md")

if __name__ == "__main__":
    main()
