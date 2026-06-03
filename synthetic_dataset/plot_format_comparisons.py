#!/usr/bin/env python3
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def get_category(song_name):
    try:
        num = int(song_name.split('_')[1])
    except:
        return "Unknown"
        
    if 1 <= num <= 10: return "Constant BPM"
    elif 11 <= num <= 20: return "60s Shifts"
    elif 21 <= num <= 30: return "40s Shifts"
    elif 31 <= num <= 40: return "50s Shifts"
    elif 41 <= num <= 50: return "Random Shifts"
    return "Unknown"

def plot_format_comparisons():
    report_files = glob.glob("*_*_*_report.json")
    if not report_files:
        print("No report JSON files found. Run the SLURM jobs first!")
        return

    # data[category][model][format] = list of errors
    categories = ["Constant BPM", "60s Shifts", "40s Shifts", "50s Shifts", "Random Shifts"]
    data = {c: {} for c in categories}
    formats = ["wav", "mp3", "ogg"]

    for file in report_files:
        basename = os.path.basename(file)
        # Expected: Model_stateless_chunk_ext_report.json
        parts = basename.split('_stateless_chunk_')
        if len(parts) != 2: continue
        model_name = parts[0]
        ext = parts[1].replace("_report.json", "")
        
        with open(file, 'r') as f:
            report = json.load(f)
            
        for song_name, error in report.items():
            if song_name == "OVERALL" or error is None: continue
            cat = get_category(song_name)
            if cat not in data: continue
            
            if model_name not in data[cat]:
                data[cat][model_name] = {f: [] for f in formats}
            
            if ext in data[cat][model_name]:
                data[cat][model_name][ext].append(error)

    # Generate 5 separate graphs (one for each category)
    for cat in categories:
        cat_data = data[cat]
        if not cat_data: continue
        
        models = list(cat_data.keys())
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, fmt in enumerate(formats):
            avg_errors = []
            for model in models:
                errors = cat_data[model].get(fmt, [])
                avg = sum(errors)/len(errors) if errors else 0
                avg_errors.append(avg)
                
            offset = width * i
            rects = ax.bar(x + offset, avg_errors, width, label=fmt.upper())
            ax.bar_label(rects, padding=3, fmt='%.1f')

        ax.set_ylabel('Average RMSE (BPM Error)')
        ax.set_title(f'Format Comparison: {cat}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        
        out_name = f"format_comparison_{cat.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        print(f"Generated {out_name}")

if __name__ == "__main__":
    plot_format_comparisons()
