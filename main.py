import argparse
import json
import os
import pandas as pd
from unsupervised import run_unsupervised_pipeline, apply_heuristics
from supervised import detect_supervised, post_process_predictions
from utils import process_point_data, interpolate_missing, calculate_derivatives

def unsupervised_hit_bounce_detection(ball_data_json):
    """
    Method 1: Unsupervised detection.
    
    Args:
        ball_data_json (dict or path): Input data.
        
    Returns:
        dict: Enriched JSON.
    """
    # If path, load df. If dict, handling is needed. 
    # Our utils expect path for loading, but we can adapt.
    # For this exercise, let's assume we receive a file path or we save the dict to a temp file if needed,
    # or better, adapt utils to load from dict. 
    # Since process_point_data takes json_path, we'll assume path for now or adapt.
    
    if isinstance(ball_data_json, dict):
        # Allow passing dict directly? 
        # For now, let's strictly follow the typical pattern where we might pass a path 
        # or we update utils to handle dict. 
        # Update: utils.load_data_from_json handles path.
        # Let's write a quick adapter if it's a dict.
        frames = []
        records = []
        for frame_id, values in ball_data_json.items():
            frames.append(int(frame_id))
            records.append(values)
        df = pd.DataFrame(records, index=frames).sort_index()
        df.index.name = 'frame'
        
        # We still need utils.interpolate_missing(df) etc.
        # Let's import the internals.
        df = interpolate_missing(df)
        df = calculate_derivatives(df)
        
        # Add rolling (Unsupervised usually relies on heuristics, 
        # but let's assume valid DF structure)
        df = apply_heuristics(df)
        
    elif isinstance(ball_data_json, str):
        df = run_unsupervised_pipeline(ball_data_json)
    else:
        raise ValueError("Input must be path (str) or data (dict)")

    # Convert back to JSON structure
    # We need to preserve original structure and add "pred_action"
    
    # Create result dict
    result = {}
    for frame, row in df.iterrows():
        frame_str = str(frame)
        # Convert numpy types to native Python types
        x_val = None if pd.isna(row['x']) else float(row['x'])
        y_val = None if pd.isna(row['y']) else float(row['y'])
        
        result[frame_str] = {
            "x": x_val,
            "y": y_val,
            "visible": bool(row['visible']) if 'visible' in row else True,
            "action": row['action'] if 'action' in row else None,
            "pred_action": row['pred_action']
        }
        
    return result

def supervised_hit_bounce_detection(ball_data_json):
    """
    Method 2: Supervised detection.
    """
    if isinstance(ball_data_json, dict):
        frames = []
        records = []
        for frame_id, values in ball_data_json.items():
            frames.append(int(frame_id))
            records.append(values)
        df = pd.DataFrame(records, index=frames).sort_index()
        df.index.name = 'frame'
        
        df = interpolate_missing(df)
        df = calculate_derivatives(df)
        
        # Rolling features
        for window in [3, 5]:
            df[f'vy_roll_mean_{window}'] = df['vy'].rolling(window=window, center=True).mean().fillna(0)
            df[f'vy_roll_std_{window}'] = df['vy'].rolling(window=window, center=True).std().fillna(0)
            df[f'speed_roll_mean_{window}'] = df['speed'].rolling(window=window, center=True).mean().fillna(0)
            
    elif isinstance(ball_data_json, str):
        df = process_point_data(ball_data_json)
    else:
        raise ValueError("Input must be path (str) or data (dict)")
        
    df = detect_supervised(df)
    df = post_process_predictions(df)

    result = {}
    for frame, row in df.iterrows():
        frame_str = str(frame)
        # Convert numpy types to native Python types
        x_val = None if pd.isna(row['x']) else float(row['x'])
        y_val = None if pd.isna(row['y']) else float(row['y'])
        
        result[frame_str] = {
            "x": x_val,
            "y": y_val,
            "visible": bool(row['visible']) if 'visible' in row else True,
            "action": row['action'] if 'action' in row else None,
            "pred_action": row['pred_action']
        }
    return result

if __name__ == "__main__":
    # Example usage
    # python main.py --input "Data hit & bounce/per_point_v2/ball_data_1.json" --method supervised
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input JSON ball data")
    parser.add_argument("--method", type=str, choices=['unsupervised', 'supervised', 'ensemble'], default='supervised')
    parser.add_argument("--output", type=str, help="Path to save output JSON", default="output.json")
    
    args = parser.parse_args()
    
    if args.input:
        if args.method == 'unsupervised':
            res = unsupervised_hit_bounce_detection(args.input)
        elif args.method == 'ensemble':
            # Ensemble currently just uses the robust supervised pipeline
            # as it now incorporates physics checks
            res = supervised_hit_bounce_detection(args.input)
        else:
            res = supervised_hit_bounce_detection(args.input)
            
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"Result saved to {args.output}")
