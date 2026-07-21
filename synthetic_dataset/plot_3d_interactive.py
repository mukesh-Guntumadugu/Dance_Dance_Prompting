import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

formats = ['wav', 'mp3', 'ogg']
runs = range(1, 11)

for ext in formats:
    print(f"Generating Interactive 3D graph for {ext.upper()}...")
    
    color_groups = {
        'Near Perfect (Error <= 0.5) [Dark Green]': {'color': 'darkgreen', 'x': [], 'y': [], 'z': []},
        'Over-predicted (0.5 < Error <= 1.0) [Cyan]': {'color': 'cyan', 'x': [], 'y': [], 'z': []},
        'Over-predicted (1.0 < Error <= 5.0) [Blue]': {'color': 'blue', 'x': [], 'y': [], 'z': []},
        'Over-predicted (Error > 5.0) [Navy]': {'color': 'navy', 'x': [], 'y': [], 'z': []},
        'Under-predicted (0.5 < Error <= 1.0) [Yellow]': {'color': '#cccc00', 'x': [], 'y': [], 'z': []}, 
        'Under-predicted (1.0 < Error <= 5.0) [Brown]': {'color': 'saddlebrown', 'x': [], 'y': [], 'z': []},
        'Under-predicted (Error > 5.0) [Red]': {'color': 'red', 'x': [], 'y': [], 'z': []},
        'Half-BPM Over-predicted (within 1) [Light Purple]': {'color': '#dca3ff', 'x': [], 'y': [], 'z': []},
        'Half-BPM Under-predicted (within 1) [Dark Purple]': {'color': '#4a0080', 'x': [], 'y': [], 'z': []},
        'Half-BPM Over-predicted (1-5 error) [Light Pink]': {'color': '#ffb3cc', 'x': [], 'y': [], 'z': []},
        'Half-BPM Under-predicted (1-5 error) [Deep Pink]': {'color': '#cc0052', 'x': [], 'y': [], 'z': []},
        'Double-BPM Over-predicted [Light Orange]': {'color': '#ffb366', 'x': [], 'y': [], 'z': []},
        'Double-BPM Under-predicted [Dark Orange]': {'color': '#cc5200', 'x': [], 'y': [], 'z': []}
    }
    
    for run in runs:
        csv_file = f'Librosa_full_song_{ext}_run{run}_rmse.csv'
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        grouped = df.groupby('actual_bpm')
        
        for actual, group in grouped:
            sq_errs = (group['pred_bpm'] - group['actual_bpm']) ** 2
            rmse = np.sqrt(sq_errs.mean())
            
            pred = group['pred_bpm'].mean()
            error = abs(actual - pred)
            
            true_half = actual / 2.0
            true_double = actual * 2.0
            
            if abs(pred - true_half) <= 0.5:
                if pred >= true_half:
                    group_name = 'Half-BPM Over-predicted (within 1) [Light Purple]'
                else:
                    group_name = 'Half-BPM Under-predicted (within 1) [Dark Purple]'
            elif abs(pred - true_half) <= 2.5:
                if pred >= true_half:
                    group_name = 'Half-BPM Over-predicted (1-5 error) [Light Pink]'
                else:
                    group_name = 'Half-BPM Under-predicted (1-5 error) [Deep Pink]'
            elif abs(pred - true_double) <= 5.0:
                if pred >= true_double:
                    group_name = 'Double-BPM Over-predicted [Light Orange]'
                else:
                    group_name = 'Double-BPM Under-predicted [Dark Orange]'
            elif error <= 0.5:
                group_name = 'Near Perfect (Error <= 0.5) [Dark Green]'
            elif pred > actual:
                if error <= 1.0:
                    group_name = 'Over-predicted (0.5 < Error <= 1.0) [Cyan]'
                elif error <= 5.0:
                    group_name = 'Over-predicted (1.0 < Error <= 5.0) [Blue]'
                else:
                    group_name = 'Over-predicted (Error > 5.0) [Navy]'
            elif pred < actual:
                if error <= 1.0:
                    group_name = 'Under-predicted (0.5 < Error <= 1.0) [Yellow]'
                elif error <= 5.0:
                    group_name = 'Under-predicted (1.0 < Error <= 5.0) [Brown]'
                else:
                    group_name = 'Under-predicted (Error > 5.0) [Red]'
            else:
                group_name = 'Under-predicted (Error > 5.0) [Red]' 
                
            color_groups[group_name]['x'].append(actual)
            color_groups[group_name]['y'].append(run) # Y is run in plotly 3D
            color_groups[group_name]['z'].append(rmse) # Z is rmse
            
    fig = go.Figure()
    
    # Plot each group
    for label, data in color_groups.items():
        if data['x']:
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=data['color'],
                    opacity=0.8,
                    line=dict(width=1, color='Black')
                ),
                name=label
            ))
            
    fig.update_layout(
        title=f'Interactive Librosa {ext.upper()} 3D RMSE Scatter Plot (Runs 1-10)',
        scene=dict(
            xaxis_title='Actual BPM',
            yaxis_title='Run Number',
            zaxis_title='RMSE',
            xaxis=dict(range=[59, 241], dtick=1),
            yaxis=dict(range=[0.5, 10.5], dtick=1),
            zaxis=dict(range=[-5, 140], dtick=1),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save direct to artifact dir
    out_name = f'/Users/mukeshguntumadugu/.gemini/antigravity-ide/brain/85734ad6-8a72-49d8-8bf4-ae795be944d2/Interactive_3D_{ext.upper()}.html'
    fig.write_html(out_name)
    print(f"Saved {out_name}")
