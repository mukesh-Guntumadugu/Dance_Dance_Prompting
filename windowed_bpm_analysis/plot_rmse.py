import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_rmse_graph():
    # Find all report JSON files
    report_files = glob.glob("*_*_report.json")
    if not report_files:
        print("No report JSON files found. Are the models still running?")
        return

    # Data structure to hold RMSE values: data[model][mode] = rmse
    data = {}
    modes_seen = set()

    for file in report_files:
        # Expected format: ModelName_mode_name_report.json
        basename = os.path.basename(file)
        # We split by '_', but 'mode' might have underscores like 'stateless_chunk'
        # So we use the known modes to parse safely.
        known_modes = ["stateless_chunk", "true_history", "fake_history", "full_song"]
        
        found_mode = None
        for m in known_modes:
            if m in basename:
                found_mode = m
                break
                
        if not found_mode:
            continue
            
        model_name = basename.split(f"_{found_mode}")[0]
        
        with open(file, 'r') as f:
            report = json.load(f)
            
        overall_rmse = report.get("OVERALL", 0)
        
        if model_name not in data:
            data[model_name] = {}
        data[model_name][found_mode] = overall_rmse
        modes_seen.add(found_mode)

    if not data:
        print("No valid data found to plot.")
        return

    # Prepare data for plotting
    models = list(data.keys())
    modes = list(modes_seen)
    
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each mode
    multiplier = 0
    for mode in modes:
        measurements = []
        for model in models:
            measurements.append(data[model].get(mode, 0.0))
            
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=mode.replace("_", " ").title())
        ax.bar_label(rects, padding=3, fmt='%.1f')
        multiplier += 1

    # Add text, title, and custom x-axis tick labels
    ax.set_ylabel('Overall RMSE (BPM)')
    ax.set_title('BPM Prediction Accuracy by Model and Context Mode')
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, max([val for m in data.values() for val in m.values()]) * 1.2) # Add 20% headroom

    plt.tight_layout()
    output_filename = "rmse_comparison_graph.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully generated graph: {output_filename}")

if __name__ == "__main__":
    generate_rmse_graph()
