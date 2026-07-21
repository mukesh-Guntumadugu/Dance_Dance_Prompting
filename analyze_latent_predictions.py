import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse

def extract_bpm_from_text(text):
    """Extract integer BPM from LLM text prediction."""
    import re
    match = re.search(r'\b(\d{2,3})\b', text)
    if match:
        return int(match.group(1))
    return 0

def analyze_model_latents(model_name, output_dir="latent_outputs"):
    print(f"Analyzing {model_name} latents...")
    
    # Load all chunks
    files = sorted(glob.glob(os.path.join(output_dir, f"{model_name}_chunk_*.pkl")))
    if not files:
        print(f"No files found for {model_name} in {output_dir}")
        return
        
    all_latents = []
    predicted_bpms = []
    chunk_indices = []
    
    for idx, f in enumerate(files):
        with open(f, "rb") as pkl:
            data = pickle.load(pkl)
            
            # Compress latents (which are [seq_len, dim]) into a single vector per chunk via mean pooling
            latents = np.array(data["latents"])
            if latents.ndim > 1:
                chunk_vector = np.mean(latents, axis=0) # [dim]
            else:
                chunk_vector = latents
                
            all_latents.append(chunk_vector)
            
            # Parse BPM
            bpm_text = str(data["bpm"])
            bpm = extract_bpm_from_text(bpm_text)
            predicted_bpms.append(bpm)
            chunk_indices.append(idx)
            
    all_latents = np.vstack(all_latents) # [num_chunks, dim]
    
    # Perform PCA down to 2 dimensions for visualization
    print(f"  Shape before PCA: {all_latents.shape}")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)
    
    # Ground Truth is 120
    ground_truth = 120
    errors = np.abs(np.array(predicted_bpms) - ground_truth)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 8))
    
    # Scatter plot, color coded by predicted BPM
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=predicted_bpms, cmap='coolwarm', s=150, edgecolor='k')
    plt.colorbar(scatter, label='Predicted BPM by LLM')
    
    # Annotate chunks
    for i, txt in enumerate(chunk_indices):
        plt.annotate(f"sec {txt}\n(pred: {predicted_bpms[i]})", (latents_2d[i, 0], latents_2d[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.title(f"{model_name.capitalize()} Latent Space vs Predicted BPM\n(Ground Truth = 120 BPM)", fontsize=14)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    # Draw an arrow showing "drift" if there is a major error (e.g. prediction != 120)
    for i, error in enumerate(errors):
        if error > 5: # Significant error
            # Find the closest correct chunk to draw a comparison arrow
            correct_indices = np.where(errors <= 5)[0]
            if len(correct_indices) > 0:
                closest_correct = correct_indices[0]
                plt.annotate('', xy=(latents_2d[i, 0], latents_2d[i, 1]), xytext=(latents_2d[closest_correct, 0], latents_2d[closest_correct, 1]),
                             arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5, width=1, headwidth=5))
                
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_img = f"latent_pattern_analysis_{model_name}.png"
    plt.savefig(out_img, dpi=300)
    print(f"  -> Saved analysis to {out_img}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="latent_outputs", help="Directory containing HPC outputs")
    args = parser.parse_args()
    
    # Analyze all models we have data for
    for model in ["qwen", "mumu", "encodec"]:
        analyze_model_latents(model, args.dir)
