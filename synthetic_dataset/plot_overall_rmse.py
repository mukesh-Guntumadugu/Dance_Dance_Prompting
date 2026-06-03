#!/usr/bin/env python3
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def get_category(song_name):
    # Extracts the number from "song_X"
    try:
        num = int(song_name.split('_')[1])
    except:
        return "Unknown"
        
    if 1 <= num <= 10:
        return "Constant BPM"
    elif 11 <= num <= 20:
        return "60s Shifts"
    elif 21 <= num <= 30:
        return "40s Shifts"
    elif 31 <= num <= 40:
        return "50s Shifts"
    elif 41 <= num <= 50:
        return "Random Shifts"
    return "Unknown"

def generate_rmse_graph():
    # Find all report JSON files
    report_files = glob.glob("*_*_report.json")
    if not report_files:
        print("No report JSON files found. Are the models still running?")
        return

    # Data structure: data[model][category] = list of errors
    data = {}
    categories = ["Constant BPM", "60s Shifts", "40s Shifts", "50s Shifts", "Random Shifts"]

    for file in report_files:
        basename = os.path.basename(file)
        model_name = basename.split("_stateless_chunk")[0]
        data[model_name] = {c: [] for c in categories}
        
        with open(file, 'r') as f:
            report = json.load(f)
            
        for song_name, error in report.items():
            if song_name == "OVERALL" or error is None:
                continue
            cat = get_category(song_name)
            if cat in data[model_name]:
                data[model_name][cat].append(error)

    if not data:
        print("No valid data found to plot.")
        return

    # Average the errors per category for each model
    avg_data = {}
    for model in data:
        avg_data[model] = []
        for cat in categories:
            errors = data[model][cat]
            avg = sum(errors) / len(errors) if errors else 0
            avg_data[model].append(avg)

    # Plot Line Graph
    models = list(avg_data.keys())
    
    x = np.arange(len(categories))  # the label locations
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in models:
        # Plot lines with markers
        ax.plot(x, avg_data[model], marker='o', linewidth=2, label=model)
        
        # Add labels to the points
        for i, val in enumerate(avg_data[model]):
            ax.annotate(f'{val:.1f}', (x[i], val), textcoords="offset points", xytext=(0,10), ha='center')

    # Add text, title, and custom x-axis tick labels
    ax.set_ylabel('Average RMSE (Error in BPM)')
    ax.set_title('BPM Prediction Error Broken Down by Song Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', ncols=len(models))
    
    # Set y-limit to max value + 20%
    max_val = max([max(vals) for vals in avg_data.values()]) if avg_data else 10
    ax.set_ylim(0, max_val * 1.2)

    plt.tight_layout()
    output_filename = "category_rmse_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully generated graph: {output_filename}")

if __name__ == "__main__":
    generate_rmse_graph()
