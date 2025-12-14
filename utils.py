import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_data_from_json(json_path):
    """
    Load a single JSON file into a pandas DataFrame.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert dict to format suitable for DataFrame
    # Input format: {"frame_num": {"x": 1, "y": 2, ...}, ...}
    frames = []
    records = []
    for frame_id, values in data.items():
        frames.append(int(frame_id))
        records.append(values)
        
    df = pd.DataFrame(records, index=frames)
    df = df.sort_index()
    # Ensure index name is frame
    df.index.name = 'frame'
    return df

def interpolate_missing(df):
    """
    Interpolate missing x, y coordinates to allow for smooth derivative calculation.
    """
    # Replace None/NaN with numpy nan
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    
    # Linear interpolation for position
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
    
    # Fill remaining NaNs (if any at start/end)
    df = df.bfill().ffill()
    return df

def calculate_derivatives(df):
    """
    Calculate velocity and acceleration features.
    Assuming constant frame rate (delta_t = 1 unit, or just proportional).
    """
    # 1st Derivative (Velocity)
    df['vx'] = df['x'].diff().fillna(0)
    df['vy'] = df['y'].diff().fillna(0)
    
    # Speed magnitude
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # 2nd Derivative (Acceleration)
    df['ax'] = df['vx'].diff().fillna(0)
    df['ay'] = df['vy'].diff().fillna(0)
    
    # Acceleration magnitude
    df['accel'] = np.sqrt(df['ax']**2 + df['ay']**2)
    
    return df

def calculate_advanced_features(df):
    """
    Calculate advanced physics-based features and temporal windows.
    """
    df = df.copy()
    
    # --- 1. Physics Features ---
    # 3rd Derivative (Jerk)
    df['jx'] = df['ax'].diff().fillna(0)
    df['jy'] = df['ay'].diff().fillna(0)
    df['jerk'] = np.sqrt(df['jx']**2 + df['jy']**2)
    
    # Angle of motion (trajectory)
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    df['angle_change'] = df['angle'].diff().fillna(0)
    # Angular velocity/curvature
    df['angle_velocity'] = df['angle_change'] # already per-frame
    
    # Velocity change magnitude (vector difference)
    df['vel_change'] = np.sqrt(df['vx'].diff().fillna(0)**2 + df['vy'].diff().fillna(0)**2)
    
    # Direction reversals (boolean -> int)
    df['vy_reversal'] = ((df['vy'] * df['vy'].shift(1)) < 0).astype(int)
    df['vx_reversal'] = ((df['vx'] * df['vx'].shift(1)) < 0).astype(int)
    
    # Speed change
    df['speed_change'] = df['speed'].diff().fillna(0)
    
    # Energy features (Kinetic Energy ~ v^2)
    df['ke'] = 0.5 * df['speed']**2
    df['ke_change'] = df['ke'].diff().fillna(0)
    
    # Absolute values
    df['ay_abs'] = df['ay'].abs()
    df['ax_abs'] = df['ax'].abs()
    
    # --- 2. Temporal Window Features (Lag/Lead) ---
    # Critical for distinguishing "hits" (future speed > past speed)
    
    # Lags (Past)
    for lag in [1, 2, 3, 5]:
        df[f'speed_lag_{lag}'] = df['speed'].shift(lag).fillna(0)
        df[f'vx_lag_{lag}'] = df['vx'].shift(lag).fillna(0)
        df[f'vy_lag_{lag}'] = df['vy'].shift(lag).fillna(0)
        df[f'accel_lag_{lag}'] = df['accel'].shift(lag).fillna(0)
        
    # Leads (Future) - Data Leakage? 
    # NO, because this is an offline processing task (or short buffer). 
    # For real-time with 0 latency it's an issue, but standard for analysis.
    # We assume we have access to the full point or a buffer.
    for lead in [1, 2, 3, 5]:
        df[f'speed_lead_{lead}'] = df['speed'].shift(-lead).fillna(0)
        df[f'vx_lead_{lead}'] = df['vx'].shift(-lead).fillna(0)
        df[f'vy_lead_{lead}'] = df['vy'].shift(-lead).fillna(0)
        df[f'accel_lead_{lead}'] = df['accel'].shift(-lead).fillna(0)
        
    # Rate of change over window (Future - Past)
    df['speed_gain_5'] = df['speed_lead_5'] - df['speed_lag_5']
    
    # Exponential moving averages
    df['speed_ema_3'] = df['speed'].ewm(span=3, adjust=False).mean()
    df['speed_ema_5'] = df['speed'].ewm(span=5, adjust=False).mean()
    
    return df

def calculate_rolling_features(df):
    """
    Calculate rolling statistics for temporal context.
    """
    # Original rolling features
    for window in [3, 5]:
        df[f'vy_roll_mean_{window}'] = df['vy'].rolling(window=window, center=True).mean().fillna(0)
        df[f'vy_roll_std_{window}'] = df['vy'].rolling(window=window, center=True).std().fillna(0)
        df[f'speed_roll_mean_{window}'] = df['speed'].rolling(window=window, center=True).mean().fillna(0)
    
    # Extended rolling features
    for window in [7, 10]:
        df[f'speed_roll_mean_{window}'] = df['speed'].rolling(window=window, center=True).mean().fillna(0)
        df[f'ay_roll_std_{window}'] = df['ay'].rolling(window=window, center=True).std().fillna(0)
        df[f'accel_roll_max_{window}'] = df['accel'].rolling(window=window, center=True).max().fillna(0)
    
    # Rolling min/max for detecting peaks
    df['y_roll_max_5'] = df['y'].rolling(window=5, center=True).max().fillna(0)
    df['y_roll_min_5'] = df['y'].rolling(window=5, center=True).min().fillna(0)
    df['is_y_local_max'] = ((df['y'] == df['y_roll_max_5']) & (df['y'] > df['y'].shift(1)) & (df['y'] > df['y'].shift(-1))).astype(int)
    
    return df

def process_point_data(json_path):
    """
    Full pipeline for loading and feature engineering a single point.
    """
    df = load_data_from_json(json_path)
    df = interpolate_missing(df)
    df = calculate_derivatives(df)
    df = calculate_advanced_features(df)
    df = calculate_rolling_features(df)
    
    return df

def load_all_data(folder_path):
    """
    Load all JSON files in the folder into a single large DataFrame.
    Adds 'point_id' column to distinguish points.
    """
    folder = Path(folder_path)
    all_dfs = []
    
    for file_path in folder.glob("*.json"):
        try:
            point_id = int(file_path.stem.split('_')[-1])  # Extract ID from ball_data_123.json
            df = load_data_from_json(file_path)
            df['point_id'] = point_id
            
            # Process features per point
            df = interpolate_missing(df)
            df = calculate_derivatives(df)
            df = calculate_advanced_features(df)
            df = calculate_rolling_features(df)
            
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            
    if not all_dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(all_dfs)
    return full_df
