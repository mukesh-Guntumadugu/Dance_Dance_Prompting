
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import datetime

def load_beatmap_measures(filepath):
    """Load beatmap measures from file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    measures = []
    current_measure = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if line == ',':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        elif len(line) == 4 and all(c in '01234M' for c in line):
            current_measure.append(line)
            
    if current_measure:
        measures.append(current_measure)
        
    return measures

def plot_vertical_chart(original_measures, generated_measures, original_label="Original", generated_label="Generated", output_file="beatmap_vertical_chart.png", start_measure=12, num_measures=8):
    """
    Plot vertical stepchart.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 16), sharey=True)
    
    # Configuration
    # We want time to go UP. 
    # Y-axis = Beats (or Measure index).
    # X-axis = Columns (0, 1, 2, 3)
    
    arrow_map = {0: '$\u2190$', 1: '$\u2193$', 2: '$\u2191$', 3: '$\u2192$'} # Left, Down, Up, Right
    col_colors = {0: '#E91E63', 1: '#00BCD4', 2: '#4CAF50', 3: '#FFC107'} # Pink, Cyan, Green, Yellow
    
    def plot_track(ax, measures, title):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(-0.5, 3.5)
        # Invert logic: Start measure at bottom, End measure at top
        ax.set_ylim(start_measure, start_measure + num_measures)
        
        # Background
        ax.set_facecolor('#222222')
        
        # Draw Column Lines
        for i in range(4):
            ax.axvline(i, color='#444444', linestyle='-', linewidth=1)
            
        # Draw Measure Lines
        for m in range(start_measure, start_measure + num_measures + 1):
            ax.axhline(m, color='white', linestyle='-', linewidth=2)
            # Add measure number text
            ax.text(-0.8, m, str(m), color='white', ha='right', va='center', fontsize=10)
            
            # Draw quarter beat lines (optional, maybe too cluttered)
            for b in range(1, 4):
                 ax.axhline(m + b/4.0, color='#666666', linestyle=':', linewidth=0.5)

        # Plot Notes
        # Iterate only through relevant measures
        end_measure = min(start_measure + num_measures, len(measures))
        
        for m_idx in range(start_measure, end_measure):
            measure = measures[m_idx]
            lines_in_measure = len(measure)
            if lines_in_measure == 0: continue
            
            # Each line is portion of a measure
            # Y position = m_idx + (line_idx / lines_in_measure)
            
            for line_idx, line in enumerate(measure):
                y_pos = m_idx + (line_idx / lines_in_measure)
                
                for col_idx, char in enumerate(line):
                    if char in '1234M':
                        # Plot Arrow
                        color = col_colors.get(col_idx, 'white')
                        rotation = 0
                        if col_idx == 0: rotation = 90
                        elif col_idx == 1: rotation = 0 # Down is implied by V shape usually, but arrow symbol needs rotation?
                        # Using matplotlib text arrows:
                        # Left arrow is usually left pointing.
                        # arrow_map has symbols.
                        
                        symbol = arrow_map.get(col_idx, '?')
                        
                        # Special handling for Mine
                        if char == 'M':
                            symbol = 'X'
                            color = 'red'
                            
                        # Hold Heads
                        if char == '2':
                            # Draw box
                            rect = patches.Rectangle((col_idx - 0.4, y_pos - 0.05), 0.8, 0.1, color=color, alpha=0.8)
                            ax.add_patch(rect)
                        
                        # Note Body
                        # Use Scatter for arrows? Or Text?
                        # Text adds symbols nicely.
                        ax.text(col_idx, y_pos, symbol, 
                               color='white', 
                               fontsize=24, 
                               ha='center', va='center', 
                               fontweight='bold',
                               bbox=dict(boxstyle="circle,pad=0.1", fc=color, ec="white", lw=1))

        # Adjust axes
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([]) # Hide labels, arrows are self explanatory
        ax.tick_params(axis='y', colors='white')
        
    # Plot Original
    plot_track(ax1, original_measures, f"Original\n{os.path.basename(original_label)}")
    
    # Plot Generated
    plot_track(ax2, generated_measures, f"Generated\n{os.path.basename(generated_label)}")
    
    # Add Date/Time Footer
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated: {timestamp}", ha="right", fontsize=10, fontstyle='italic', color='white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, facecolor='#222222')
    print(f"Vertical chart saved to {output_file}")

def main():
    # Configuration
    GEN_FILE = "generated_MechaTribe_eassy_20260211_012048.txt"
    ORIG_FILE = "MechaTribe_original_easy.ssc.text"
    
    # Load
    print("Loading beatmaps...")
    gen_measures = load_beatmap_measures(GEN_FILE)
    orig_measures = load_beatmap_measures(ORIG_FILE)
    
    # Choose a section with notes
    # Start at measure 12 (approx where song starts after intro?)
    # or measure 24 (approx 48s / (60/180 * 4) = 48 / 1.33 = 36 measures?)
    # Song starts at 48s -> 180 BPM -> 3 beats per sec -> 144 beats -> 36 measures.
    # Actually BPM is 180.
    # 180 beats / 60 sec = 3 beats/sec.
    # 48 sec * 3 = 144 beats.
    # 144 beats / 4 = 36 measures.
    
    START_MEASURE = 36
    NUM_MEASURES = 8
    
    plot_vertical_chart(orig_measures, gen_measures, 
                       original_label=ORIG_FILE, 
                       generated_label=GEN_FILE,
                       start_measure=START_MEASURE, num_measures=NUM_MEASURES)

if __name__ == "__main__":
    main()
