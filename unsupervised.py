import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from utils import process_point_data

def apply_heuristics(df, 
                     ay_percentile=0.90,
                     accel_percentile=0.90,
                     min_speed_for_hit=2.0):
    """
    Improved unsupervised detection with physics-based heuristics.
    Leverages smoothed coordinates for cleaner derivatives.
    Phase 3 Logic (Optimal).
    """
    df = df.copy()
    df['pred_action'] = 'air'
    
    # --- Adaptive Thresholds ---
    # Stricter thresholds (90th percentile) to improve precision
    ay_thresh = df['ay'].abs().quantile(ay_percentile)
    accel_thresh = df['accel'].quantile(accel_percentile)
    
    # --- 1. BOUNCE DETECTION ---
    # Logic: Local Maxima in Y AND (Accel Spike OR Velocity Reversal)
    
    # Find local maxima (peaks in Y) - corresponding to ground contact
    y_values = df['y'].values
    # Increased distance to prevent finding peaks too close to each other
    peaks, _ = find_peaks(y_values, prominence=5, distance=10)
    
    for p in peaks:
        idx = df.index[p]
        
        # Check window for physics confirmation
        loc_idx = df.index.get_loc(idx)
        start = max(0, loc_idx - 3)
        end = min(len(df), loc_idx + 4) # Look +3 frames ahead
        window = df.iloc[start:end]
        
        # Condition A: Vertical Acceleration Spike
        has_accel_spike = window['ay'].abs().max() > ay_thresh
        
        # Condition B: Strict V-Shape (Velocity Reversal)
        # We need a clear transition from +vy (down) to -vy (up)
        # Sum of vy before should be positive, sum after should be negative
        vy_in = window.iloc[:3]['vy'].mean() 
        vy_out = window.iloc[-3:]['vy'].mean()
        
        is_v_shape = (vy_in > 0.5) and (vy_out < -0.5)
        
        # Combine: Peak + (Strict Reversal OR Strong Accel Spike)
        if is_v_shape or (has_accel_spike and (vy_in > 0 and vy_out < 0)):
            df.loc[idx, 'pred_action'] = 'bounce'

    # --- 2. HIT DETECTION ---
    # Logic: High Accel AND (Speed Increase OR Trajectory Reset)
    
    # Candidates: High acceleration events (excluding already marked bounces)
    candidates = df[df['accel'] > accel_thresh].index
    
    for idx in candidates:
        if df.loc[idx, 'pred_action'] == 'bounce':
            continue
            
        # Check window
        loc_idx = df.index.get_loc(idx)
        start = max(0, loc_idx - 2)
        end = min(len(df), loc_idx + 3)
        window = df.iloc[start:end]
        
        if len(window) < 4:
            continue

        # Feature 1: Speed Increase (Racket adds energy)
        speed_before = window['speed'].iloc[0]
        speed_after = window['speed'].iloc[-1]
        has_energy_gain = speed_after > speed_before * 1.10  # 10% increase required now
        
        # Feature 2: Trajectory Reset (Dot Product)
        # If velocity vectors before and after oppose each other or change significantly
        v_in = np.array([window['vx'].iloc[0], window['vy'].iloc[0]])
        v_out = np.array([window['vx'].iloc[-1], window['vy'].iloc[-1]])
        
        # Normalize
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        
        if norm_in > 0.1 and norm_out > 0.1:
            dot_prod = np.dot(v_in, v_out) / (norm_in * norm_out)
            # If dot product is low (< 0.5), angle is > 60 degrees. 
            # If dot product is negative, direction reversed.
            has_trajectory_change = dot_prod < 0.7 # Significant change
        else:
            has_trajectory_change = False
        
        # Condition D: Minimum Activity
        is_moving = df.loc[idx, 'speed'] > min_speed_for_hit
        
        if is_moving and (has_energy_gain or has_trajectory_change):
             df.loc[idx, 'pred_action'] = 'hit'
    # --- Post-Processing ---
    # Robust Non-Maximum Suppression (NMS)
    # Group consecutive detections and keep only the strongest one
    
    # 1. Get all candidate indices
    candidates = df[df['pred_action'] != 'air'].index.tolist()
    if not candidates:
        return df
        
    # 2. Cluster candidates (events within 10 frames of each other)
    clusters = []
    current_cluster = [candidates[0]]
    
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i-1] <= 10:
            current_cluster.append(candidates[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [candidates[i]]
    clusters.append(current_cluster)
    
    # 3. Select best candidate per cluster
    final_indices = []
    final_actions = []
    
    for cluster in clusters:
        # Determine dominanat action type in cluster (bounce vs hit)
        # Priority: Bounce > Hit (Bounces are more distinct in Y)
        actions = df.loc[cluster, 'pred_action']
        if 'bounce' in actions.values:
            best_action = 'bounce'
            # For bounce, pick the frame with local max Y (highest visible point? NO. Lowest visual point = Max Y)
            # Actually, we rely on the one with highest vertical acceleration spike or just center.
            # Let's pick max 'accel' as impact is highest there.
            cluster_subset = df.loc[cluster]
            winner_idx = cluster_subset['accel'].idxmax()
        else:
            best_action = 'hit'
            # For hit, pick max accel
            cluster_subset = df.loc[cluster]
            winner_idx = cluster_subset['accel'].idxmax()
            
        final_indices.append(winner_idx)
        final_actions.append(best_action)
        
    # 4. Apply cleanup
    df['pred_action'] = 'air' # Reset
    df.loc[final_indices, 'pred_action'] = final_actions
    
    return df

def remove_isolated_events(df, min_gap=5):
    """
    Deprecated in Phase 2 favored of raw detection, 
    but kept as placeholder if needed later.
    """
    return df

def run_unsupervised_pipeline(json_path):
    """
    Run improved unsupervised detection pipeline.
    """
    df = process_point_data(json_path)
    df = apply_heuristics(df)
    return df
