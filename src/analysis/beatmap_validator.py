"""
Beatmap Validation System

This module provides comprehensive validation of beatmap step placements against
musical features including onsets, beats, and percussive elements.

Features:
1. Onset alignment validation - check if steps are placed on musical onsets
2. Beat alignment validation - check if steps align with detected beats
3. Feature-based placement - validate steps against drums/percussive elements
4. Comparison mode - compare original vs generated beatmaps
5. Detailed reporting - JSON, text, and visualizations
"""

import numpy as np
import librosa
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


@dataclass
class ValidationResults:
    """Container for validation results"""
    # Onset alignment
    onset_alignment_percentage: float
    steps_on_onsets: int
    total_steps: int
    mean_onset_distance_ms: float
    
    # Beat alignment
    beat_alignment_percentage: float
    steps_on_beats: int
    mean_beat_distance_ms: float
    
    # Feature-based placement
    percussive_alignment_percentage: float
    steps_on_percussive: int
    
    # Additional metrics
    onset_times: List[float]
    step_times: List[float]
    beat_times: List[float]
    distances_to_onsets: List[float]
    distances_to_beats: List[float]
    
    # Metadata
    audio_file: str
    beatmap_file: str
    bpm: float
    offset: float
    timestamp: str


def load_beatmap_measures(filepath: str) -> List[List[str]]:
    """
    Load beatmap measures from file.
    
    Args:
        filepath: Path to beatmap file (.txt or .text)
        
    Returns:
        List of measures, where each measure is a list of note rows
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    measures = []
    current_measure = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line == ',':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        elif len(line) == 4 and all(c in '01234M' for c in line):
            current_measure.append(line)
            
    if current_measure:
        measures.append(current_measure)
        
    return measures


def get_step_times(measures: List[List[str]], bpm: float, offset: float) -> np.ndarray:
    """
    Convert beatmap measures to timestamps.
    
    Args:
        measures: List of beatmap measures
        bpm: Beats per minute
        offset: Time offset in seconds where first beat occurs
        
    Returns:
        Array of timestamps for each step
    """
    step_times = []
    beats_per_measure = 4  # Assuming 4/4 time signature
    seconds_per_beat = 60.0 / bpm
    
    start_time = offset
    
    for measure_idx, measure in enumerate(measures):
        measure_start_beat = measure_idx * beats_per_measure
        lines_in_measure = len(measure)
        if lines_in_measure == 0:
            continue
        
        beats_per_line = beats_per_measure / lines_in_measure
        
        for line_idx, line in enumerate(measure):
            # Check if there is a note (any non-zero, excluding mines for now)
            has_note = any(c in '1234' for c in line)
            
            if has_note:
                beat_time = measure_start_beat + (line_idx * beats_per_line)
                timestamp = start_time + (beat_time * seconds_per_beat)
                step_times.append(timestamp)
                
    return np.array(step_times)


def detect_onsets(audio_path: str, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Detect musical onsets in audio.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Additional arguments for librosa.onset.onset_detect
        
    Returns:
        Tuple of (onset_times, sample_rate)
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Detect onsets with backtracking for better accuracy
    onset_frames = librosa.onset.onset_detect(
        y=y, 
        sr=sr, 
        backtrack=True,
        **kwargs
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    return onset_times, sr


def detect_beats(audio_path: str) -> Tuple[np.ndarray, float]:
    """
    Detect beats in audio.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (beat_times, estimated_tempo)
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Handle tempo - it can be an array or scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    
    return beat_times, tempo


def extract_percussive_onsets(audio_path: str) -> np.ndarray:
    """
    Extract onsets from percussive component of audio.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Array of percussive onset times
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Detect onsets on percussive component
    onset_frames = librosa.onset.onset_detect(
        y=y_percussive,
        sr=sr,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    return onset_times


def calculate_alignment(
    step_times: np.ndarray,
    reference_times: np.ndarray,
    tolerance_ms: float = 50.0
) -> Tuple[float, int, List[float]]:
    """
    Calculate alignment percentage between steps and reference times.
    
    Args:
        step_times: Array of step timestamps
        reference_times: Array of reference timestamps (onsets/beats)
        tolerance_ms: Tolerance window in milliseconds
        
    Returns:
        Tuple of (alignment_percentage, aligned_count, distances_list)
    """
    if len(step_times) == 0 or len(reference_times) == 0:
        return 0.0, 0, []
    
    tolerance_sec = tolerance_ms / 1000.0
    aligned_count = 0
    distances = []
    
    for step_time in step_times:
        # Find nearest reference time
        idx = np.searchsorted(reference_times, step_time)
        
        min_distance = float('inf')
        
        # Check left neighbor
        if idx > 0:
            left_dist = abs(step_time - reference_times[idx - 1])
            min_distance = min(min_distance, left_dist)
        
        # Check right neighbor
        if idx < len(reference_times):
            right_dist = abs(step_time - reference_times[idx])
            min_distance = min(min_distance, right_dist)
        
        distances.append(min_distance)
        
        # Check if within tolerance
        if min_distance <= tolerance_sec:
            aligned_count += 1
    
    alignment_percentage = (aligned_count / len(step_times)) * 100.0
    
    return alignment_percentage, aligned_count, distances


def validate_beatmap(
    audio_path: str,
    beatmap_path: str,
    bpm: float,
    offset: float,
    tolerance_ms: float = 50.0
) -> ValidationResults:
    """
    Validate beatmap step placements against musical features.
    
    Args:
        audio_path: Path to audio file
        beatmap_path: Path to beatmap file
        bpm: Beats per minute
        offset: Time offset in seconds
        tolerance_ms: Tolerance window for alignment (default 50ms)
        
    Returns:
        ValidationResults object with all metrics
    """
    print(f"\n{'='*60}")
    print(f"BEATMAP VALIDATION")
    print(f"{'='*60}")
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Beatmap: {os.path.basename(beatmap_path)}")
    print(f"BPM: {bpm}, Offset: {offset}s")
    print(f"Tolerance: {tolerance_ms}ms")
    
    # Load beatmap and get step times
    print("\n[1/5] Loading beatmap...")
    measures = load_beatmap_measures(beatmap_path)
    step_times = get_step_times(measures, bpm, offset)
    print(f"  Found {len(step_times)} steps")
    
    # Detect onsets
    print("\n[2/5] Detecting onsets...")
    onset_times, sr = detect_onsets(audio_path)
    print(f"  Detected {len(onset_times)} onsets")
    
    # Detect beats
    print("\n[3/5] Detecting beats...")
    beat_times, estimated_tempo = detect_beats(audio_path)
    print(f"  Detected {len(beat_times)} beats (tempo: {estimated_tempo:.1f} BPM)")
    
    # Extract percussive onsets
    print("\n[4/5] Extracting percussive features...")
    percussive_onset_times = extract_percussive_onsets(audio_path)
    print(f"  Found {len(percussive_onset_times)} percussive onsets")
    
    # Calculate alignments
    print("\n[5/5] Calculating alignments...")
    
    # Onset alignment
    onset_align_pct, steps_on_onsets, onset_distances = calculate_alignment(
        step_times, onset_times, tolerance_ms
    )
    mean_onset_dist_ms = np.mean(onset_distances) * 1000.0 if onset_distances else 0.0
    
    # Beat alignment
    beat_align_pct, steps_on_beats, beat_distances = calculate_alignment(
        step_times, beat_times, tolerance_ms
    )
    mean_beat_dist_ms = np.mean(beat_distances) * 1000.0 if beat_distances else 0.0
    
    # Percussive alignment
    perc_align_pct, steps_on_percussive, perc_distances = calculate_alignment(
        step_times, percussive_onset_times, tolerance_ms
    )
    
    # Create results object
    results = ValidationResults(
        onset_alignment_percentage=onset_align_pct,
        steps_on_onsets=steps_on_onsets,
        total_steps=len(step_times),
        mean_onset_distance_ms=mean_onset_dist_ms,
        
        beat_alignment_percentage=beat_align_pct,
        steps_on_beats=steps_on_beats,
        mean_beat_distance_ms=mean_beat_dist_ms,
        
        percussive_alignment_percentage=perc_align_pct,
        steps_on_percussive=steps_on_percussive,
        
        onset_times=onset_times.tolist(),
        step_times=step_times.tolist(),
        beat_times=beat_times.tolist(),
        distances_to_onsets=onset_distances,
        distances_to_beats=beat_distances,
        
        audio_file=audio_path,
        beatmap_file=beatmap_path,
        bpm=bpm,
        offset=offset,
        timestamp=datetime.now().isoformat()
    )
    
    return results


def compare_beatmaps(
    audio_path: str,
    original_beatmap_path: str,
    generated_beatmap_path: str,
    bpm: float,
    offset: float,
    tolerance_ms: float = 50.0
) -> Tuple[ValidationResults, ValidationResults]:
    """
    Compare original and generated beatmaps.
    
    Args:
        audio_path: Path to audio file
        original_beatmap_path: Path to original beatmap
        generated_beatmap_path: Path to generated beatmap
        bpm: Beats per minute
        offset: Time offset in seconds
        tolerance_ms: Tolerance window for alignment
        
    Returns:
        Tuple of (original_results, generated_results)
    """
    print("\n" + "="*60)
    print("COMPARING BEATMAPS")
    print("="*60)
    
    print("\n>>> Validating ORIGINAL beatmap...")
    original_results = validate_beatmap(
        audio_path, original_beatmap_path, bpm, offset, tolerance_ms
    )
    
    print("\n>>> Validating GENERATED beatmap...")
    generated_results = validate_beatmap(
        audio_path, generated_beatmap_path, bpm, offset, tolerance_ms
    )
    
    return original_results, generated_results


def print_validation_report(results: ValidationResults):
    """
    Print a comprehensive validation report.
    
    Args:
        results: ValidationResults object
    """
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    
    print(f"\n📊 OVERVIEW")
    print(f"  Total Steps: {results.total_steps}")
    print(f"  BPM: {results.bpm}")
    print(f"  Offset: {results.offset}s")
    
    print(f"\n🎵 ONSET ALIGNMENT")
    print(f"  Percentage on Onsets: {results.onset_alignment_percentage:.2f}%")
    print(f"  Steps on Onsets: {results.steps_on_onsets}/{results.total_steps}")
    print(f"  Mean Distance to Nearest Onset: {results.mean_onset_distance_ms:.2f}ms")
    
    # Rating
    if results.onset_alignment_percentage >= 80:
        rating = "✅ EXCELLENT"
    elif results.onset_alignment_percentage >= 60:
        rating = "✓ GOOD"
    elif results.onset_alignment_percentage >= 40:
        rating = "⚠ FAIR"
    else:
        rating = "❌ POOR"
    print(f"  Rating: {rating}")
    
    print(f"\n🥁 BEAT ALIGNMENT")
    print(f"  Percentage on Beats: {results.beat_alignment_percentage:.2f}%")
    print(f"  Steps on Beats: {results.steps_on_beats}/{results.total_steps}")
    print(f"  Mean Distance to Nearest Beat: {results.mean_beat_distance_ms:.2f}ms")
    
    # Rating
    if results.beat_alignment_percentage >= 80:
        rating = "✅ EXCELLENT"
    elif results.beat_alignment_percentage >= 60:
        rating = "✓ GOOD"
    elif results.beat_alignment_percentage >= 40:
        rating = "⚠ FAIR"
    else:
        rating = "❌ POOR"
    print(f"  Rating: {rating}")
    
    print(f"\n🔊 PERCUSSIVE ALIGNMENT")
    print(f"  Percentage on Percussive Features: {results.percussive_alignment_percentage:.2f}%")
    print(f"  Steps on Percussive: {results.steps_on_percussive}/{results.total_steps}")
    
    # Rating
    if results.percussive_alignment_percentage >= 70:
        rating = "✅ EXCELLENT"
    elif results.percussive_alignment_percentage >= 50:
        rating = "✓ GOOD"
    elif results.percussive_alignment_percentage >= 30:
        rating = "⚠ FAIR"
    else:
        rating = "❌ POOR"
    print(f"  Rating: {rating}")
    
    print(f"\n{'='*60}")


def print_comparison_report(
    original: ValidationResults,
    generated: ValidationResults
):
    """
    Print a comparison report between original and generated beatmaps.
    
    Args:
        original: Validation results for original beatmap
        generated: Validation results for generated beatmap
    """
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")
    
    print(f"\n📊 OVERVIEW")
    print(f"  Original Steps: {original.total_steps}")
    print(f"  Generated Steps: {generated.total_steps}")
    print(f"  Difference: {generated.total_steps - original.total_steps:+d}")
    
    print(f"\n🎵 ONSET ALIGNMENT COMPARISON")
    print(f"  Original:  {original.onset_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.onset_alignment_percentage:6.2f}%")
    diff = generated.onset_alignment_percentage - original.onset_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\n🥁 BEAT ALIGNMENT COMPARISON")
    print(f"  Original:  {original.beat_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.beat_alignment_percentage:6.2f}%")
    diff = generated.beat_alignment_percentage - original.beat_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\n🔊 PERCUSSIVE ALIGNMENT COMPARISON")
    print(f"  Original:  {original.percussive_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.percussive_alignment_percentage:6.2f}%")
    diff = generated.percussive_alignment_percentage - original.percussive_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\n{'='*60}")


def save_results_json(results: ValidationResults, output_path: str):
    """
    Save validation results to JSON file.
    
    Args:
        results: ValidationResults object
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


def visualize_validation(
    results: ValidationResults,
    output_path: str = "validation_visualization.png"
):
    """
    Create comprehensive visualization of validation results.
    
    Args:
        results: ValidationResults object
        output_path: Path to save visualization
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1])
    
    # 1. Timeline visualization (Top, full width)
    ax1 = plt.subplot(gs[0, :])
    
    # Plot onsets
    onset_times = np.array(results.onset_times)
    ax1.scatter(onset_times, np.ones_like(onset_times) * 3, 
                marker='|', s=100, c='gray', alpha=0.5, label='Onsets')
    
    # Plot beats
    beat_times = np.array(results.beat_times)
    ax1.scatter(beat_times, np.ones_like(beat_times) * 2,
                marker='|', s=150, c='blue', alpha=0.6, label='Beats')
    
    # Plot steps
    step_times = np.array(results.step_times)
    ax1.scatter(step_times, np.ones_like(step_times) * 1,
                marker='o', s=50, c='red', alpha=0.8, label='Steps')
    
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Steps', 'Beats', 'Onsets'])
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_title(f'Timeline Visualization - {os.path.basename(results.beatmap_file)}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Onset distance histogram
    ax2 = plt.subplot(gs[1, 0])
    distances_ms = np.array(results.distances_to_onsets) * 1000
    ax2.hist(distances_ms, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.axvline(50, color='red', linestyle='--', linewidth=2, label='50ms threshold')
    ax2.set_xlabel('Distance to Nearest Onset (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Onset Distance Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Beat distance histogram
    ax3 = plt.subplot(gs[1, 1])
    beat_distances_ms = np.array(results.distances_to_beats) * 1000
    ax3.hist(beat_distances_ms, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(50, color='red', linestyle='--', linewidth=2, label='50ms threshold')
    ax3.set_xlabel('Distance to Nearest Beat (ms)')
    ax3.set_ylabel('Count')
    ax3.set_title('Beat Distance Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Alignment percentages bar chart
    ax4 = plt.subplot(gs[2, :])
    categories = ['Onset\nAlignment', 'Beat\nAlignment', 'Percussive\nAlignment']
    values = [
        results.onset_alignment_percentage,
        results.beat_alignment_percentage,
        results.percussive_alignment_percentage
    ]
    colors = ['#9C27B0', '#2196F3', '#FF9800']
    
    bars = ax4.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add threshold line
    ax4.axhline(80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (80%)')
    ax4.axhline(60, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (60%)')
    
    ax4.set_ylabel('Alignment Percentage (%)', fontsize=12)
    ax4.set_title('Alignment Metrics Summary', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Beatmap Validator Module")
    print("Import this module and use validate_beatmap() or compare_beatmaps()")
    print("\nExample:")
    print("  from beatmap_validator import validate_beatmap, print_validation_report")
    print("  results = validate_beatmap('song.mp3', 'beatmap.txt', bpm=180, offset=-0.028)")
    print("  print_validation_report(results)")
