import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from utils import process_point_data

def apply_heuristics(df, 
                     ay_percentile=0.70,
                     accel_percentile=0.70,
                     y_bounce_percentile=0.90,
                     min_speed_for_hit=5.0):
    """
    Improved unsupervised detection with adaptive thresholds and better peak detection.
    """
    df = df.copy()
    df['pred_action'] = 'air'
    
    # --- Adaptive Thresholds ---
    # Relaxed thresholds to improve recall
    ay_thresh = df['ay'].abs().quantile(ay_percentile)
    accel_thresh = df['accel'].quantile(accel_percentile)
    
    # Bounce height requirements (bounces happen on ground = high Y in screen coords)
    y_ground_thresh = df['y'].quantile(y_bounce_percentile)
    
    # --- 1. BOUNCE DETECTION ---
    # Logic: Local Maxima in Y AND (Accel Spike OR Velocity Reversal)
    
    # Find local maxima (peaks in Y)
    # Invert Y for find_peaks because Y increases downwards in screen coords? 
    # Actually, usually Y=0 is top. So ground is HIGH Y. 
    # So we want peaks (maxima) in Y directly.
    y_values = df['y'].values
    peaks, _ = find_peaks(y_values, prominence=10, distance=5)
    
    for p in peaks:
        idx = df.index[p]
        
        # Condition A: Position (must be in lower part of screen)
        if df.loc[idx, 'y'] < y_ground_thresh:
            continue
            
        # Check window for physics confirmation
        start = max(0, p - 3)
        end = min(len(df), p + 3)
        window = df.iloc[start:end]
        
        # Condition B: Vertical Acceleration Spike
        has_accel_spike = window['ay'].abs().max() > ay_thresh
        
        # Condition C: Velocity Reversal (vy changes sign)
        # Check if vy goes from positive (down) to negative (up)
        # We look for sign change in window
        has_reversal = (window['vy'].min() < 0) and (window['vy'].max() > 0)
        
        # Combine: Peak + (Accel OR Reversal)
        if has_accel_spike or has_reversal:
            df.loc[idx, 'pred_action'] = 'bounce'

    # --- 2. HIT DETECTION ---
    # Logic: High Accel AND (Speed Increase OR Vx Reversal)
    
    # Candidates: High acceleration events (excluding already marked bounces)
    candidates = df[df['accel'] > accel_thresh].index
    
    for idx in candidates:
        if df.loc[idx, 'pred_action'] == 'bounce':
            continue
            
        # Check window
        loc_idx = df.index.get_loc(idx)
        start = max(0, loc_idx - 2)
        end = min(len(df), loc_idx + 2)
        window = df.iloc[start:end]
        
        # Condition A: Speed Increase (Racket adds energy)
        # Compare max speed after event vs min speed before/at event
        speed_before = window['speed'].iloc[0]
        speed_after = window['speed'].iloc[-1]
        has_energy_gain = speed_after > speed_before * 1.1  # 10% increase
        
        # Condition B: Horizontal Direction Change (Return/Volley)
        has_vx_reversal = (window['vx'].min() < 0) and (window['vx'].max() > 0)
        
        # Condition C: Minimum Activity (Ball must be moving)
        is_moving = df.loc[idx, 'speed'] > min_speed_for_hit
        
        if is_moving and (has_energy_gain or has_vx_reversal):
             df.loc[idx, 'pred_action'] = 'hit'
             
    # --- Post-Processing ---
    # Simple clean up of consecutive events - keep max confidence
    # For unsupervised, we assume max accel is the 'true' event center
    
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
