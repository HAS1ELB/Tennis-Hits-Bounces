"""
Quick evaluation script to check model performance on a few sample points.
"""
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from main import unsupervised_hit_bounce_detection, supervised_hit_bounce_detection

def evaluate_method(method_name, method_func, test_files):
    """Evaluate a detection method on test files."""
    all_true = []
    all_pred = []
    
    for file_path in test_files:
        result = method_func(file_path)
        
        for frame_id, data in result.items():
            if data['action'] is not None:  # Only evaluate labeled frames
                all_true.append(data['action'])
                all_pred.append(data['pred_action'])
    
    report = classification_report(all_true, all_pred, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=['air', 'bounce', 'hit'])
    
    print("\n" + "="*40 + "\nCONSOLE OUTPUT:\n" + report + "\n" + "="*40 + "\n")
    
    with open("evaluation_report.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{method_name} Results\n")
        f.write(f"{'='*60}\n")
        f.write(report + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f"{'='*60}\n\n")
        
    print(f"Results for {method_name} written to evaluation_report.txt")

if __name__ == "__main__":
    from glob import glob
    import os
    
    # Locate all JSON files
    data_dir = "Data hit & bounce/per_point_v2"
    test_files = glob(os.path.join(data_dir, "*.json"))
    
    print(f"Found {len(test_files)} files for evaluation.")
    
    # Run evaluation
    if test_files:
        print("Evaluating UNSUPERVISED METHOD on all files...")
        evaluate_method("UNSUPERVISED METHOD", unsupervised_hit_bounce_detection, test_files)
        
        print("Evaluating SUPERVISED METHOD on all files...")
        evaluate_method("SUPERVISED METHOD", supervised_hit_bounce_detection, test_files)
    else:
        print(f"No files found in {data_dir}")
