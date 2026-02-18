#!/usr/bin/env python3
"""
Example script demonstrating beatmap validation.

This script shows how to use the beatmap_validator module to:
1. Validate a single beatmap against audio
2. Compare original vs generated beatmaps
3. Generate reports and visualizations
"""

import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.beatmap_validator import (
    validate_beatmap,
    compare_beatmaps,
    print_validation_report,
    print_comparison_report,
    save_results_json,
    visualize_validation
)


def example_single_validation():
    """Example: Validate a single beatmap"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Beatmap Validation")
    print("="*70)
    
    # Configuration - Update these paths for your files
    audio_path = "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3"
    beatmap_path = "src/musicForBeatmap/Springtime/beatmap_easy.txt"
    bpm = 142.0
    offset = -0.028
    
    # Validate
    results = validate_beatmap(
        audio_path=audio_path,
        beatmap_path=beatmap_path,
        bpm=bpm,
        offset=offset,
        tolerance_ms=50.0  # 50ms tolerance window
    )
    
    # Print report
    print_validation_report(results)
    
    # Save JSON
    save_results_json(results, "validation_results.json")
    
    # Create visualization
    visualize_validation(results, "validation_visualization.png")
    
    return results


def example_comparison():
    """Example: Compare original vs generated beatmap"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Original vs Generated Comparison")
    print("="*70)
    
    # Configuration - Update these paths
    audio_path = "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3"
    original_beatmap = "src/musicForBeatmap/Springtime/beatmap_easy.txt"
    generated_beatmap = "generated_springtime_beatmap_20260210_010532.text"
    bpm = 142.0
    offset = -0.028
    
    # Check if generated beatmap exists
    if not os.path.exists(generated_beatmap):
        print(f"\n⚠ Generated beatmap not found: {generated_beatmap}")
        print("Skipping comparison example.")
        return None, None
    
    # Compare
    original_results, generated_results = compare_beatmaps(
        audio_path=audio_path,
        original_beatmap_path=original_beatmap,
        generated_beatmap_path=generated_beatmap,
        bpm=bpm,
        offset=offset,
        tolerance_ms=50.0
    )
    
    # Print comparison report
    print_comparison_report(original_results, generated_results)
    
    # Save results
    save_results_json(original_results, "original_validation.json")
    save_results_json(generated_results, "generated_validation.json")
    
    # Create visualizations
    visualize_validation(original_results, "original_validation_viz.png")
    visualize_validation(generated_results, "generated_validation_viz.png")
    
    return original_results, generated_results


def example_custom_tolerance():
    """Example: Test different tolerance values"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Testing Different Tolerance Values")
    print("="*70)
    
    audio_path = "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3"
    beatmap_path = "src/musicForBeatmap/Springtime/beatmap_easy.txt"
    bpm = 142.0
    offset = -0.028
    
    tolerances = [30, 50, 75, 100]  # milliseconds
    
    print("\nTesting tolerance values:", tolerances, "ms")
    print("\n" + "-"*70)
    
    for tol in tolerances:
        results = validate_beatmap(
            audio_path, beatmap_path, bpm, offset, tolerance_ms=tol
        )
        
        print(f"\nTolerance: {tol}ms")
        print(f"  Onset Alignment: {results.onset_alignment_percentage:.1f}%")
        print(f"  Beat Alignment:  {results.beat_alignment_percentage:.1f}%")
        print(f"  Percussive:      {results.percussive_alignment_percentage:.1f}%")


def main():
    """Main function to run examples"""
    print("\n" + "="*70)
    print("BEATMAP VALIDATOR - EXAMPLE USAGE")
    print("="*70)
    
    # Check if required files exist
    springtime_audio = "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3"
    springtime_beatmap = "src/musicForBeatmap/Springtime/beatmap_easy.txt"
    
    if not os.path.exists(springtime_audio):
        print(f"\n❌ Error: Audio file not found: {springtime_audio}")
        print("Please update the file paths in this script to match your setup.")
        return
    
    if not os.path.exists(springtime_beatmap):
        print(f"\n❌ Error: Beatmap file not found: {springtime_beatmap}")
        print("Please update the file paths in this script to match your setup.")
        return
    
    # Run examples
    try:
        # Example 1: Single validation
        results = example_single_validation()
        
        # Example 2: Comparison (if generated beatmap exists)
        orig, gen = example_comparison()
        
        # Example 3: Different tolerances
        example_custom_tolerance()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  - validation_results.json")
        print("  - validation_visualization.png")
        if orig and gen:
            print("  - original_validation.json")
            print("  - generated_validation.json")
            print("  - original_validation_viz.png")
            print("  - generated_validation_viz.png")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
