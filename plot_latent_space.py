import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def plot_latent_space(csv_path, output_path):
    print(f"Reading encoded tokens from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop the Frame column so we only have the layers
    if "Frame" in df.columns:
        layers = df.drop(columns=["Frame"])
    else:
        layers = df
        
    # The dataframe is currently [Time x Layers]. Let's transpose it to [Layers x Time] for the heatmap
    matrix = layers.to_numpy().T
    
    # Let's just look at the first 3 seconds to make the visualization clear.
    # EnCodec runs at 75 frames per second. 3 seconds = 225 frames.
    frames_per_sec = 75
    time_limit = 3 * frames_per_sec
    matrix = matrix[:, :time_limit]
    
    plt.figure(figsize=(15, 6))
    
    # Create the heatmap
    sns.heatmap(matrix, cmap="viridis", cbar_kws={'label': 'Token Value (0-1023)'})
    
    plt.title("Internal Encoded Music Representation (First 3 Seconds)", fontsize=16)
    plt.ylabel("Codebook Layers (Depth)", fontsize=12)
    plt.xlabel("Time (Frames)", fontsize=12)
    
    # Add second X-axis for Seconds
    ax = plt.gca()
    ax_sec = ax.twiny()
    
    # Set the limits of the second axis to match the frames
    ax_sec.set_xlim(0, time_limit / frames_per_sec)
    ax_sec.set_xlabel("Time (Seconds)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="src/Neural Audio Codecs/outputs/test_slice_20260701_092206_tokens.csv")
    parser.add_argument("--out", default="encoded_heatmap.png")
    args = parser.parse_args()
    
    plot_latent_space(args.csv, args.out)
