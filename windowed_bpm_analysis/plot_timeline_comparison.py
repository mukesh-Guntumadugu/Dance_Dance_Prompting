import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt

def generate_timeline_graphs():
    # 1. Find all CSV files for a specific mode (let's default to stateless_chunk)
    mode = "stateless_chunk"
    csv_files = glob.glob(f"*_{mode}_rmse.csv")
    
    if not csv_files:
        print(f"No CSV files found for mode {mode}.")
        return

    # 2. Load all the data
    model_data = {}
    all_songs = set()
    
    for f in csv_files:
        model_name = os.path.basename(f).split(f"_{mode}")[0]
        try:
            # Drop NUL bytes if file was actively being written
            with open(f, 'r') as file:
                content = file.read().replace('\0', '')
            
            # Read from clean string
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            
            if len(df) == 0 or 'pred_bpm' not in df.columns:
                continue
                
            model_data[model_name] = df
            all_songs.update(df['song_name'].unique())
        except Exception as e:
            print(f"Error reading {f}: {e}")

    all_songs = sorted(list(all_songs))
    if not all_songs:
        print("No valid songs found in the CSVs.")
        return

    # 3. Create a giant figure with a subplot for every single song
    num_songs = len(all_songs)
    cols = min(2, num_songs)  # 2 columns max for readability
    rows = math.ceil(num_songs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, song in enumerate(all_songs):
        ax = axes[i]
        title_rmses = []
        
        # Plot the ACTUAL true BPM first (as a thick black line)
        # We can get the actual BPM from any of the models since it's the same
        first_model = list(model_data.keys())[0]
        song_df = model_data[first_model][model_data[first_model]['song_name'] == song]
        
        # Plot true BPM using a step plot because BPM holds steady across the window
        ax.step(song_df['window_start'], song_df['actual_bpm'], where='post', 
                color='black', linewidth=3, label='True/Actual BPM', zorder=5)

        # 4. Now plot EVERY model's prediction on top, and calculate its RMSE for this song
        for color_idx, (model_name, df) in enumerate(model_data.items()):
            m_song_df = df[df['song_name'] == song]
            if len(m_song_df) == 0:
                continue
                
            # Filter out the 0.0 failed predictions for the plot
            valid_df = m_song_df[m_song_df['pred_bpm'] > 0]
            if len(valid_df) == 0:
                title_rmses.append(f"{model_name}: N/A")
                continue

            # Calculate RMSE for this song
            rmse = math.sqrt(((valid_df['pred_bpm'] - valid_df['actual_bpm'])**2).mean())
            title_rmses.append(f"{model_name}: {rmse:.1f}")
            
            # Plot model predictions
            c = colors[color_idx % len(colors)]
            ax.step(valid_df['window_start'], valid_df['pred_bpm'], where='post', 
                    color=c, linewidth=2, alpha=0.7, linestyle='--', label=f'{model_name} Prediction')

        # Format the subplot
        ax.set_title(f"{song}\nRMSE -> " + " | ".join(title_rmses), fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("BPM")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')

    # Remove any empty subplots if the grid isn't perfectly full
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_filename = f"timeline_comparison_{mode}.png"
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    print(f"\nSuccessfully generated giant timeline graph: {output_filename}")


if __name__ == "__main__":
    generate_timeline_graphs()
