import matplotlib.pyplot as plt
import os

def render_local_chart():
    # Hardcoded metadata cleanly parsed from the live Ohio HPC SSH terminal stdout
    chunks = [20, 15, 10, 5, 2]
    hits = [206, 222, 350, 363, 400]
    f1_scores = [36.1, 36.1, 43.2, 43.7, 41.8]
    
    labels = [f"{c}s Segment" for c in chunks]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # F1 Score Bars
    color = 'tab:blue'
    ax1.set_xlabel('Processing Audio Vector Size', fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', color=color, fontweight='bold')
    bars = ax1.bar(labels, f1_scores, color=color, alpha=0.7, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 50) 

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', color=color)

    # Clean True Positives Scatter/Line
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Exact Hits Detected (Count)', color=color2, fontweight='bold')
    line = ax2.plot(labels, hits, color=color2, marker='o', markersize=8, linewidth=2, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 450)

    for i, txt in enumerate(hits):
        ax2.annotate(f"{txt} hits", (labels[i], hits[i] + 15), textcoords="offset points", xytext=(0,5), ha='center', color=color2, fontweight='bold')

    plt.title('LLM Onset Detection Stability by Temporal Bounds\nSong: Bad Ketchup (Tolerance: ±50ms)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    out_file = os.path.join("analysis_reports", "Qwen_Onset_Sweep_Comparison_Bad_Ketchup.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Generated native graph securely at {out_file} without requiring SSH keys!")

if __name__ == "__main__":
    render_local_chart()
