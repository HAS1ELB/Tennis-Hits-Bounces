import pandas as pd
import numpy as np
import json
import os
from supervised import detect_supervised
from utils import process_point_data
import joblib
from supervised import MODEL_PATH, get_enhanced_features_and_labels

def analyze_false_positives(test_files, output_file="false_positive_report.csv"):
    """
    Analyze false positives in the test set.
    """
    results = []
    
    # Load model for probability extraction
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found.")
        return

    print("Analyzing false positives...")
    
    for json_path in test_files:
        filename = os.path.basename(json_path)
        point_id = filename.split('.')[0]
        
        # Load and process
        df = process_point_data(json_path)
        
        # Get predictions
        df_pred = detect_supervised(df)
        
        # Get raw probabilities
        X, _ = get_enhanced_features_and_labels(df)
        if hasattr(X, 'values'):
             X = X.values
        probs = clf.predict_proba(X)
        
        # Identify False Positives
        # FP: Predicted event (Hit/Bounce) but Ground Truth is Air
        # Note: We might also care about misclassified events (Hit predicted as Bounce), 
        # but "False Positive" strictly usually means "Alarm when no event".
        # The user defined it as "12 total false positives ... out of 2554 air frames", 
        # so specifically Air classified as Event.
        
        fp_mask = (df_pred['pred_action'].isin(['hit', 'bounce'])) & (df_pred['action'] == 'air')
        fp_indices = df_pred[fp_mask].index
        
        for idx in fp_indices:
            row = df.loc[idx]
            pred = df_pred.loc[idx, 'pred_action']
            
            # Find context (features)
            # Probability
            loc_idx = df.index.get_loc(idx)
            prob_bounce = probs[loc_idx, 1]
            prob_hit = probs[loc_idx, 2]
            
            # Check proximity to real event
            # Find closest GT event
            gt_events = df[df['action'] != 'air'].index
            min_dist = 999
            nearest_event = "None"
            if len(gt_events) > 0:
                dists = [abs(idx - gt_idx) for gt_idx in gt_events]
                min_dist = min(dists)
                nearest_event = df.loc[gt_events[np.argmin(dists)], 'action']
            
            results.append({
                'point_id': point_id,
                'frame': idx,
                'ground_truth': 'air',
                'predicted': pred,
                'prob_bounce': round(prob_bounce, 3),
                'prob_hit': round(prob_hit, 3),
                'accel': round(row['accel'], 2),
                'speed': round(row['speed'], 2),
                'y': round(row['y'], 2),
                'vy': round(row['vy'], 2),
                'dist_to_real_event': min_dist,
                'nearest_real_event': nearest_event
            })
            
    # Save results
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_file, index=False)
        print(f"False positive analysis saved to {output_file}")
        print(df_res)
    else:
        print("No false positives found in the provided files.")

if __name__ == "__main__":
    # Use the same sample files as evaluate.py for consistency in checking the "12 false positives"
    test_files = [
        "Data hit & bounce/per_point_v2/ball_data_1.json",
        "Data hit & bounce/per_point_v2/ball_data_10.json",
        "Data hit & bounce/per_point_v2/ball_data_100.json"
    ]
    analyze_false_positives(test_files)
