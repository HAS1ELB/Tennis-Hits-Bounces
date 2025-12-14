import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from utils import load_all_data
from supervised import get_enhanced_features_and_labels, post_process_predictions

def run_cross_validation(data_folder="Data hit & bounce/per_point_v2", k_folds=5, output_file="cv_results.csv"):
    """
    Perform K-Fold Cross Validation on the supervised model.
    """
    print(f"Starting {k_folds}-Fold Cross Validation...")
    
    # 1. Load Data
    df = load_all_data(data_folder)
    if df.empty:
        print("No data found!")
        return

    # Clean data
    df = df.dropna(subset=['action'])
    point_ids = df['point_id'].unique()
    
    print(f"Total points: {len(point_ids)}")
    print(f"Total frames: {len(df)}")
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = []
    fold = 0
    
    # Aggregate confusion matrix
    total_cm = np.zeros((3, 3), dtype=int)
    
    for train_idx, test_idx in kf.split(point_ids):
        fold += 1
        print(f"\n--- Fold {fold}/{k_folds} ---")
        
        train_points = point_ids[train_idx]
        test_points = point_ids[test_idx]
        
        train_df = df[df['point_id'].isin(train_points)].reset_index(drop=True)
        test_df = df[df['point_id'].isin(test_points)].reset_index(drop=True)
        
        # Feature Extraction
        X_train, y_train = get_enhanced_features_and_labels(train_df)
        X_test, y_test = get_enhanced_features_and_labels(test_df)
        
        # SMOTE Balancing (Training only)
        # print("  Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train.astype(float), y_train)
        
        # Train Model
        # print("  Training Model...")
        label_map = {'air': 0, 'bounce': 1, 'hit': 2}
        y_train_encoded = y_train_resampled.map(label_map)
        
        clf = xgb.XGBClassifier(
            n_estimators=200, # Reduced slightly for speed during CV
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            n_jobs=-1,
            random_state=42
        )
        
        clf.fit(X_train_resampled.values, y_train_encoded.values)
        
        # Prediction
        probs = clf.predict_proba(X_test.values)
        
        # Inference with Thresholds (Same as supervised.py)
        bounce_thresh = 0.4
        hit_thresh = 0.3
        
        # Create temp DF for post-processing
        fold_pred_df = test_df.copy()
        fold_pred_df['pred_action'] = 'air'
        
        bounce_mask = probs[:, 1] > bounce_thresh
        hit_mask = probs[:, 2] > hit_thresh
        
        fold_pred_df.loc[hit_mask, 'pred_action'] = 'hit'
        fold_pred_df.loc[bounce_mask & (~hit_mask), 'pred_action'] = 'bounce'
        
        # Post-Processing
        fold_pred_df = post_process_predictions(fold_pred_df)
        
        # Evaluation
        y_true = fold_pred_df['action']
        y_pred = fold_pred_df['pred_action']
        
        # Metrics
        labels = ['air', 'bounce', 'hit']
        
        # Per-class F1
        f1_MACRO = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        total_cm += cm
        
        print(f"  Macro F1: {f1_MACRO:.4f}")
        print(f"  Recall (Air/Bounce/Hit): {recalls}")
        print(f"  Prec (Air/Bounce/Hit):   {precisions}")
        
        res = {
            'fold': fold,
            'macro_f1': f1_MACRO,
            'bounce_recall': recalls[1],
            'hit_recall': recalls[2],
            'bounce_prec': precisions[1],
            'hit_prec': precisions[2],
            'bounce_f1': f1s[1],
            'hit_f1': f1s[2]
        }
        results.append(res)
        
    # Summary
    res_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("CROSS-VALIDATION SUMMARY")
    print("="*40)
    print(res_df.mean())
    print("\nAggregate Confusion Matrix:")
    print(total_cm)
    
    res_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    run_cross_validation()
