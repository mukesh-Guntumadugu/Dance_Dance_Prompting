#!/usr/bin/env python3
"""
Beatmap Alignment Visualizer
Creates an interactive HTML timeline showing audio onsets/beats
and overlays the step placements from generated beatmaps.

Usage:
  python3 src/analysis/visualize_alignment.py \
    --audio "path/to/song.ogg" \
    --beatmaps "path/to/gemini.txt" "path/to/qwen.txt" "path/to/original.txt" \
    --output "alignment_compare.html"
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import plotly.graph_objects as go
from src.analysis.beatmap_validator import detect_onsets, detect_beats, load_beatmap_measures, get_step_times

# Color palette for different beatmaps
COLORS = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

def visualize_alignment(audio_path, beatmap_paths, output_html, bpm=120.0, offset=0.0):
    print(f"Visualizing alignment to: {os.path.basename(audio_path)}")
    
    # 1. Detect Audio Features
    print("Detecting onsets...")
    onset_times, sr = detect_onsets(audio_path)
    print(f"  Found {len(onset_times)} onsets")
    
    print("Detecting beats...")
    beat_times, tempo = detect_beats(audio_path, start_bpm=bpm)
    print(f"  Found {len(beat_times)} beats (estimated tempo: {tempo:.1f})")

    fig = go.Figure()

    # 2. Add Audio Features as background lines
    fig.add_trace(go.Scatter(
        x=onset_times,
        y=[-1] * len(onset_times),
        mode='markers',
        marker=dict(symbol='line-ns-open', size=15, color='rgba(200, 200, 200, 0.4)', line=dict(width=2)),
        name='Audio Onsets',
        hoverinfo='x',
        hovertemplate="Onset: %{x:.3f}s<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=beat_times,
        y=[-2] * len(beat_times),
        mode='markers',
        marker=dict(symbol='line-ns-open', size=15, color='rgba(150, 150, 255, 0.5)', line=dict(width=2)),
        name='Audio Beats',
        hoverinfo='x',
        hovertemplate="Beat: %{x:.3f}s<extra></extra>"
    ))

    # 3. Process each Beatmap
    for i, bp_path in enumerate(beatmap_paths):
        if not os.path.exists(bp_path):
            print(f"⚠️ Warning: Missing beatmap {bp_path}")
            continue
            
        name = os.path.basename(bp_path).replace('.txt', '').replace('.csv', '')
        # Shorten giant generated names for the legend
        if "gemini-pro" in name or "gemini-3" in name:
            display_name = f"Gemini ({name.split('_')[-1]})"
        elif "Qwen" in name:
            display_name = f"Qwen ({name.split('_')[-2]})"
        elif "original" in name.lower() or ".ssc" in bp_path.lower():
            display_name = "Original Chart"
        else:
            display_name = name[:20]

        print(f"Processing {display_name}...")
        measures = load_beatmap_measures(bp_path, "Hard")
        step_times, total_n, indices, contents, partials = get_step_times(measures, bpm, offset)
        
        y_pos = i  # Stack them correctly
        
        # Add as scatter points
        fig.add_trace(go.Scatter(
            x=step_times,
            y=[y_pos] * len(step_times),
            mode='markers',
            marker=dict(
                symbol='circle',
                size=10, 
                color=COLORS[i % len(COLORS)],
                line=dict(color='white', width=1)
            ),
            name=display_name,
            text=[f"Row: {c}<br>Snap: {p}" for c, p in zip(contents, partials)],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Time: %{x:.3f}s<br>"
                "%{text}"
                "<extra></extra>"
            )
        ))

    # 4. Configure Layout
    fig.update_layout(
        title="Beatmap Sync Alignment (Zoom in to see exact ms offsets)",
        xaxis=dict(
            title="Time (seconds)",
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title="Beatmap",
            tickvals=list(range(-2, len(beatmap_paths))),
            ticktext=['Audio Beats', 'Audio Onsets'] + [os.path.basename(p)[:15] for p in beatmap_paths],
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        hovermode="x unified",
        height=400 + (len(beatmap_paths) * 100),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # Save
    fig.write_html(output_html)
    print(f"\n✅ Visualization built! Open {output_html} in your browser.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AI Beatmap Alignment")
    parser.add_argument("--audio", required=True, help="Path to original audio (.ogg/.mp3)")
    parser.add_argument("--beatmaps", nargs='*', help="Paths to generated beatmap files")
    parser.add_argument("--pattern", type=str, help="Substring to find in filename (e.g. task0003)")
    parser.add_argument("--output", default="alignment_compare.html", help="HTML output file name")
    parser.add_argument("--bpm", type=float, default=120.0, help="Fallback BPM if metadata is missing")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset in seconds")
    
    args = parser.parse_args()
    
    beatmaps = args.beatmaps or []
    if args.pattern:
        song_dir = os.path.dirname(os.path.abspath(args.audio))
        for f in os.listdir(song_dir):
            path = os.path.join(song_dir, f)
            # Find original chart
            if f.endswith(('.ssc', '.sm')) and path not in beatmaps:
                beatmaps.append(path)
            # Find matching generated txts
            elif args.pattern in f and f.endswith('.txt') and path not in beatmaps:
                beatmaps.append(path)
                
    if not beatmaps:
        print("Error: No beatmaps provided or found matching pattern.")
        sys.exit(1)
        
    visualize_alignment(args.audio, beatmaps, args.output, args.bpm, args.offset)
