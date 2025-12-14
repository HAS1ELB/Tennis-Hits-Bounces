import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Import project utilities and models
from utils import process_point_data
from unsupervised import run_unsupervised_pipeline
from supervised import detect_supervised

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_point_diagnosis(json_path, output_dir="diagnostics/plots"):
    """
    Generate diagnostic plots for a single point.
    """
    ensure_dir(output_dir)
    filename = Path(json_path).stem
    
    # Load and process data
    df = process_point_data(json_path)
    
    # Get predictions
    df = detect_supervised(df)
    supervised_preds = df['pred_action'].copy()
    
    df_un = run_unsupervised_pipeline(json_path)
    unsupervised_preds = df_un['pred_action'].copy()
    
    # --- PLOT 1: Trajectory with Events ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot path
    visible_df = df[df['visible'] == True]
    ax.plot(visible_df['x'], visible_df['y'], 'o-', markersize=2, alpha=0.3, label='Ball Path', color='gray')
    
    # Invert Y to match screen coordinates (0 is top)
    ax.invert_yaxis()
    
    # Plot Ground Truth
    gt_bounces = df[df['action'] == 'bounce']
    gt_hits = df[df['action'] == 'hit']
    
    ax.scatter(gt_bounces['x'], gt_bounces['y'], c='green', s=100, marker='^', label='GT Bounce', zorder=5)
    ax.scatter(gt_hits['x'], gt_hits['y'], c='blue', s=100, marker='*', label='GT Hit', zorder=5)
    
    # Plot Predictions (Supervised)
    sup_bounces = df[supervised_preds == 'bounce']
    sup_hits = df[supervised_preds == 'hit']
    
    ax.scatter(sup_bounces['x'], sup_bounces['y'], facecolors='none', edgecolors='red', s=150, linewidth=2, marker='o', label='Pred Bounce (Sup)', zorder=4)
    ax.scatter(sup_hits['x'], sup_hits['y'], facecolors='none', edgecolors='orange', s=150, linewidth=2, marker='s', label='Pred Hit (Sup)', zorder=4)
    
    ax.set_title(f"Trajectory Diagnosis: {filename}")
    ax.legend()
    plt.savefig(f"{output_dir}/{filename}_trajectory.png")
    plt.close()
    
    # --- PLOT 2: Physics Features over Time ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 2.1 Vertical Motion
    ax1.plot(df.index, df['y'], label='Y Position', color='black', alpha=0.5)
    ax1.plot(df.index, df['vy']*10, label='Vy * 10', color='blue', alpha=0.5)
    ax1.set_ylabel("Position / Velocity")
    ax1.set_title("Vertical Motion (Detection of Bounces)")
    ax1.invert_yaxis() # Match screen
    
    # Highlight GT Bounces
    for frame in gt_bounces.index:
        ax1.axvline(x=frame, color='green', alpha=0.5, linestyle='--')
        ax1.text(frame, df.loc[frame, 'y'], 'GT Hit', rotation=90, verticalalignment='bottom')

    # 2.2 Horizontal Motion & Speed
    ax2.plot(df.index, df['speed'], label='Speed', color='purple')
    ax2.plot(df.index, df['vx'].abs(), label='|Vx|', color='orange', linestyle='--')
    ax2.set_ylabel("Speed")
    ax2.set_title("Speed & Horizontal Motion (Detection of Hits)")
    
    # Highlight GT Hits
    for frame in gt_hits.index:
        ax2.axvline(x=frame, color='blue', alpha=0.5, linestyle='--')
        
    # 2.3 Acceleration
    ax3.plot(df.index, df['accel'], label='Accel Magnitude', color='red')
    ax3.plot(df.index, df['ay'].abs(), label='|Ay|', color='pink', linestyle='--')
    ax3.set_ylabel("Acceleration")
    ax3.set_title("Acceleration Profile")
    
    # Threshold lines (Visual reference)
    ax3.axhline(y=df['accel'].quantile(0.95), color='gray', linestyle=':', label='95th Percentile')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_physics.png")
    plt.close()

def analyze_detection_failures(json_path, method="unsupervised", output_file="diagnostics/diagnostics_summary.txt"):
    """
    Analyze why false negatives and false positives occurred.
    """
    df = process_point_data(json_path)
    filename = Path(json_path).stem
    
    if method == "unsupervised":
        preds_df = run_unsupervised_pipeline(json_path)
        preds = preds_df['pred_action']
    else:
        preds_df = detect_supervised(df)
        preds = preds_df['pred_action']
        
    result_str = []
    result_str.append(f"\nAnalysis for {filename} ({method}):")
    result_str.append("-" * 40)
    
    # 1. Analyze False Negatives (Missed GT events)
    gt_events = df[df['action'].isin(['bounce', 'hit'])]
    
    if gt_events.empty:
        result_str.append("No ground truth events in this file.")
    else:
        for frame, row in gt_events.iterrows():
            actual = row['action']
            predicted = preds.loc[frame]
            
            result_str.append(f"Frame {frame}: GT={actual}, Pred={predicted}")
            
            if actual != predicted:
                result_str.append(f"  [MISS] Context:")
                result_str.append(f"    y={row['y']:.1f}, vy={row['vy']:.1f}, speed={row['speed']:.1f}")
                result_str.append(f"    ay={row['ay']:.1f}, accel={row['accel']:.1f}")
                
                # Check quantile ranks to see if it was 'significant'
                ay_rank = (df['ay'].abs() < abs(row['ay'])).mean()
                accel_rank = (df['accel'] < row['accel']).mean()
                result_str.append(f"    Global Ranks: |ay| > {ay_rank:.2%}, accel > {accel_rank:.2%}")
                
    # 2. Analyze False Positives
    fp_mask = (preds.isin(['bounce', 'hit'])) & (df['action'] == 'air')
    fp_frames = df[fp_mask].index
    
    if len(fp_frames) > 0:
        result_str.append(f"\nFalse Positives ({len(fp_frames)} frames):")
        for frame in fp_frames[:5]: # limit output
            pred = preds.loc[frame]
            row = df.loc[frame]
            result_str.append(f"  Frame {frame}: Pred={pred} (GT=air)")
            result_str.append(f"    ay={row['ay']:.1f}, accel={row['accel']:.1f}, speed={row['speed']:.1f}")
            
    # Write to file and print
    final_output = "\n".join(result_str)
    print(final_output)
    
    with open(output_file, "a") as f:
        f.write(final_output + "\n")

if __name__ == "__main__":
    # Test on a few files
    test_files = [
        "Data hit & bounce/per_point_v2/ball_data_1.json",
        "Data hit & bounce/per_point_v2/ball_data_10.json",
        "Data hit & bounce/per_point_v2/ball_data_100.json"
    ]
    
    # Clear previous summary
    if os.path.exists("diagnostics/diagnostics_summary.txt"):
        os.remove("diagnostics/diagnostics_summary.txt")
        
    for f in test_files:
        print(f"Diagnosing {f}...")
        visualize_point_diagnosis(f)
        analyze_detection_failures(f, method="supervised")
        analyze_detection_failures(f, method="unsupervised")
