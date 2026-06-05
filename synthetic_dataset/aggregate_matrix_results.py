import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def main():
    matrix_dir = "matrix_dataset"
    models = ["Qwen", "Flamingo", "MuMu", "Librosa", "DeepResonance"]
    
    # Store aggregated results: {model: {base_bpm: rmse}}
    results = {m: {} for m in models}
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(matrix_dir, "*_rmse.csv"))
    
    for f in csv_files:
        basename = os.path.basename(f)
        # e.g. Qwen_base_bpm_60_rmse.csv
        m = re.match(r"([a-zA-Z]+)_base_bpm_(\d+)_rmse\.csv", basename)
        if m:
            model = m.group(1)
            base_bpm = int(m.group(2))
            
            # Ensure model is valid
            if model not in models:
                continue
                
            try:
                df = pd.read_csv(f)
                if len(df) == 0:
                    continue
                # Calculate RMSE for this base_bpm
                df["error_sq"] = (df["actual_bpm"] - df["pred_bpm"]) ** 2
                rmse = np.sqrt(df["error_sq"].mean())
                results[model][base_bpm] = rmse
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    # Create output directory for plots
    os.makedirs("analysis_reports", exist_ok=True)
    
    # Create heatmaps
    plt.figure(figsize=(15, 10))
    
    # We expect BPMs from 60 to 240
    bpms = list(range(60, 241))
    
    for i, model in enumerate(models):
        plt.subplot(2, 3, i+1)
        
        # We'll plot a 1D heatmap (a line or a strip)
        # Or better, just a line plot for RMSE across base_bpm
        model_results = results[model]
        if not model_results:
            plt.title(f"{model} (No Data)")
            plt.axis('off')
            continue
            
        x = sorted(model_results.keys())
        y = [model_results[k] for k in x]
        
        plt.plot(x, y, marker='o', linestyle='-', markersize=4)
        plt.title(f"{model} (N={len(x)})")
        plt.xlabel("Base BPM")
        plt.ylabel("RMSE")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 50) # Cap Y axis for better comparison
        
    plt.tight_layout()
    plt.savefig("analysis_reports/matrix_rmse_comparison.png")
    print("Saved plot to analysis_reports/matrix_rmse_comparison.png")
    
    # Also write a summary text report
    with open("analysis_reports/matrix_summary.txt", "w") as f:
        f.write("Matrix Evaluation Summary\n")
        f.write("=========================\n\n")
        for model in models:
            model_results = results[model]
            f.write(f"Model: {model}\n")
            f.write(f"Files Completed: {len(model_results)} / 181\n")
            if model_results:
                avg_rmse = np.mean(list(model_results.values()))
                f.write(f"Average RMSE across completed files: {avg_rmse:.2f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
