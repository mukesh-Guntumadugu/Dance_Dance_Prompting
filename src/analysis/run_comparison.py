
import sys
import os

# Add src to python path to import generation module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.beatmap_comparison import load_beatmap, calculate_all_metrics

def main():
    gen_file = "generated_MechaTribe_eassy_20260211_012048.txt"
    orig_file = "MechaTribe_original_easy.ssc.text"
    
    # Check if files exist
    if not os.path.exists(gen_file):
        print(f"Error: Generated file not found at {gen_file}")
        return
    if not os.path.exists(orig_file):
        print(f"Error: Original file not found at {orig_file}")
        return

    print(f"Loading beatmaps...")
    print(f"Generated: {gen_file}")
    print(f"Original: {orig_file}")
    
    generated = load_beatmap(gen_file)
    original = load_beatmap(orig_file)
    
    print(f"\nGenerated lines: {len(generated)}")
    print(f"Original lines: {len(original)}")
    
    print(f"\nCalculating metrics...")
    metrics = calculate_all_metrics(generated, original)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON METRICS")
    print(f"{'='*60}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
    print(f"Timing Accuracy (±2 lines): {metrics['timing_accuracy']:.2f}%")
    print(f"Density Accuracy: {metrics['density_accuracy']:.2f}%")
    print(f"\nPer-Direction Metrics:")
    for direction, scores in metrics['per_direction'].items():
        print(f"  {direction.capitalize()}: P={scores['precision']:.1f}% R={scores['recall']:.1f}% F1={scores['f1_score']:.1f}%")
    print(f"{'='*60}")
    
    # Additional insights
    print("\nAditional Insights:")
    print(f"Generated Total Notes: {metrics['generated_total_notes']}")
    print(f"Original Total Notes: {metrics['original_total_notes']}")
    print(f"Density Ratio: {metrics['density_ratio']:.2f}")


if __name__ == "__main__":
    main()
