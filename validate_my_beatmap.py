#!/usr/bin/env python3
"""
Simple Beatmap Validator Script

INSTRUCTIONS:
1. Edit the CONFIGURATION section below with your file paths
2. Run: python3 validate_my_beatmap.py
3. Check the output files generated
"""

import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from analysis.beatmap_validator import (
    validate_beatmap,
    compare_beatmaps,
    print_validation_report,
    print_comparison_report,
    save_results_json,
    visualize_validation
)


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION WITH YOUR FILE PATHS
# ============================================================================

# Required: Audio file path (mp3, ogg, wav)
AUDIO_FILE = "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3"

# Required: Beatmap file to validate (.txt or .text format)
BEATMAP_FILE = "src/musicForBeatmap/Springtime/beatmap_easy.txt"

# Optional: Generated/comparison beatmap (leave as None if not comparing)
GENERATED_BEATMAP_FILE = "generated_springtime_beatmap_20260210_010532.text"
# Set to None to skip comparison:
# GENERATED_BEATMAP_FILE = None

# Required: Song metadata
BPM = 142.0              # Beats per minute (from .ssc file)
OFFSET = -0.028          # Time offset in seconds (from .ssc file)

# Optional: Tolerance window in milliseconds (default: 50ms)
TOLERANCE_MS = 50.0

# Output file paths (will be created in the same directory as this script)
OUTPUT_JSON = "my_validation_results.json"
OUTPUT_VISUALIZATION = "my_validation_visualization.png"

# If comparing beatmaps, output files for comparison
COMPARISON_ORIGINAL_JSON = "my_original_validation.json"
COMPARISON_GENERATED_JSON = "my_generated_validation.json"
COMPARISON_ORIGINAL_VIZ = "my_original_viz.png"
COMPARISON_GENERATED_VIZ = "my_generated_viz.png"

# ============================================================================
# END CONFIGURATION - DO NOT EDIT BELOW THIS LINE
# ============================================================================


def main():
    """Run beatmap validation based on configuration above"""
    
    print("\n" + "="*70)
    print("BEATMAP VALIDATOR")
    print("="*70)
    
    # Check if required files exist
    if not os.path.exists(AUDIO_FILE):
        print(f"\n❌ Error: Audio file not found: {AUDIO_FILE}")
        print("Please update AUDIO_FILE in the configuration section.")
        return
    
    if not os.path.exists(BEATMAP_FILE):
        print(f"\n❌ Error: Beatmap file not found: {BEATMAP_FILE}")
        print("Please update BEATMAP_FILE in the configuration section.")
        return
    
    # Validate main beatmap
    print(f"\n{'='*70}")
    print("VALIDATING BEATMAP")
    print(f"{'='*70}")
    
    results = validate_beatmap(
        audio_path=AUDIO_FILE,
        beatmap_path=BEATMAP_FILE,
        bpm=BPM,
        offset=OFFSET,
        tolerance_ms=TOLERANCE_MS
    )
    
    # Print report
    print_validation_report(results)
    
    # Save outputs
    save_results_json(results, OUTPUT_JSON)
    visualize_validation(results, OUTPUT_VISUALIZATION)
    
    print(f"\n✅ Validation complete!")
    print(f"\n📁 Generated files:")
    print(f"   - {OUTPUT_JSON}")
    print(f"   - {OUTPUT_VISUALIZATION}")
    
    # Compare with generated beatmap if specified
    if GENERATED_BEATMAP_FILE and GENERATED_BEATMAP_FILE != "None":
        if not os.path.exists(GENERATED_BEATMAP_FILE):
            print(f"\n⚠ Warning: Generated beatmap not found: {GENERATED_BEATMAP_FILE}")
            print("Skipping comparison.")
        else:
            print(f"\n{'='*70}")
            print("COMPARING BEATMAPS")
            print(f"{'='*70}")
            
            original_results, generated_results = compare_beatmaps(
                audio_path=AUDIO_FILE,
                original_beatmap_path=BEATMAP_FILE,
                generated_beatmap_path=GENERATED_BEATMAP_FILE,
                bpm=BPM,
                offset=OFFSET,
                tolerance_ms=TOLERANCE_MS
            )
            
            # Print comparison
            print_comparison_report(original_results, generated_results)
            
            # Save comparison outputs
            save_results_json(original_results, COMPARISON_ORIGINAL_JSON)
            save_results_json(generated_results, COMPARISON_GENERATED_JSON)
            visualize_validation(original_results, COMPARISON_ORIGINAL_VIZ)
            visualize_validation(generated_results, COMPARISON_GENERATED_VIZ)
            
            print(f"\n✅ Comparison complete!")
            print(f"\n📁 Additional comparison files:")
            print(f"   - {COMPARISON_ORIGINAL_JSON}")
            print(f"   - {COMPARISON_GENERATED_JSON}")
            print(f"   - {COMPARISON_ORIGINAL_VIZ}")
            print(f"   - {COMPARISON_GENERATED_VIZ}")
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
