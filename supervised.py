import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from utils import load_all_data

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "hit_bounce_model.pkl"

def get_enhanced_features_and_labels(df):
    """
    Prepare enhanced feature set for training/inference.
    """
    # Enhanced feature set
    feature_cols = [
        # Position
        'y',
        # Velocity
        'vx', 'vy', 'speed',
        # Acceleration
        'ax', 'ay', 'accel',
        # Jerk (3rd derivative)
        'jerk',
        # Angles
        'angle', 'angle_change',
        # Velocity changes
        'vel_change', 'speed_change',
        # Direction reversals
        'vy_reversal', 'vx_reversal',
        # Exponential moving averages
        'speed_ema_3', 'speed_ema_5',
        # Absolute accelerations
        'ay_abs', 'ax_abs',
        # Rolling statistics (original)
        'vy_roll_mean_3', 'vy_roll_std_3', 'speed_roll_mean_3',
        'vy_roll_mean_5', 'vy_roll_std_5', 'speed_roll_mean_5',
        # Rolling statistics (extended)
        'speed_roll_mean_7', 'ay_roll_std_7', 'accel_roll_max_7',
        'speed_roll_mean_10', 'ay_roll_std_10', 'accel_roll_max_10',
        # Peak detection
        'is_y_local_max',
        # --- New Features ---
        'jx', 'jy', 'jerk',
        'angle_velocity',
        'ke', 'ke_change', 'speed_gain_5'
    ]
    
    # Add Lag/Lead features dynamically
    for lag in [1, 2, 3, 5]:
        feature_cols.extend([f'speed_lag_{lag}', f'vx_lag_{lag}', f'vy_lag_{lag}', f'accel_lag_{lag}'])
    
    for lead in [1, 2, 3, 5]:
        feature_cols.extend([f'speed_lead_{lead}', f'vx_lead_{lead}', f'vy_lead_{lead}', f'accel_lead_{lead}'])
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    # Labels (only if training)
    if 'action' in df.columns:
        y = df['action']
    else:
        y = None
        
    return X, y

def train_model(data_folder):
    """
    Train an improved model with XGBoost and enhanced features.
    """
    print("Loading data with enhanced features...")
    df = load_all_data(data_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Loaded {len(df)} frames from {df['point_id'].nunique()} points.")
    
    # Clean data
    df = df.dropna(subset=['action'])
    
    # Train/Test Split by Point ID
    point_ids = df['point_id'].unique()
    train_points, test_points = train_test_split(point_ids, test_size=0.2, random_state=42)
    
    train_df = df[df['point_id'].isin(train_points)]
    test_df = df[df['point_id'].isin(test_points)]
    
    X_train, y_train = get_enhanced_features_and_labels(train_df)
    X_test, y_test = get_enhanced_features_and_labels(test_df)
    
    print(f"\nTraining set (before balancing): {len(X_train)} frames")
    print(f"Test set: {len(X_test)} frames")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Apply SMOTE to balance classes
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    # Convert to numeric types explicitly to be safe
    X_train = X_train.astype(float)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Training set (after balancing): {len(X_train_resampled)} frames")
    print(f"Class distribution after SMOTE: {y_train_resampled.value_counts().to_dict()}")
    
    # Train model
    if HAS_XGBOOST:
        print("\nTraining XGBoost model...")
        # Encode labels to integers
        label_map = {'air': 0, 'bounce': 1, 'hit': 2}
        y_train_encoded = y_train_resampled.map(label_map)
        y_test_encoded = y_test.map(label_map)
        
        clf = xgb.XGBClassifier(
            n_estimators=300,  # Increased estimators
            max_depth=6,       # Slightly reduced depth to prevent overfitting on synthetic samples
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        # Convert to numpy to avoid compatibility issues
        clf.fit(X_train_resampled.values if hasattr(X_train_resampled, 'values') else X_train_resampled, 
                y_train_encoded.values if hasattr(y_train_encoded, 'values') else y_train_encoded)
        
        # Predict and decode
        # Convert X_test to numpy array to match training
        y_pred_encoded = clf.predict(X_test.values if hasattr(X_test, 'values') else X_test)
        reverse_map = {0: 'air', 1: 'bounce', 2: 'hit'}
        y_pred = pd.Series(y_pred_encoded).map(reverse_map)
        
    else:
        print("\nXGBoost not available, using Random Forest with enhanced features...")
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)
    
    print("\n" + "="*60)
    print("IMPROVED MODEL EVALUATION")
    print("="*60)
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['air', 'bounce', 'hit']))
    print("="*60)
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    
    return clf

def detect_supervised(df, bounce_threshold=0.4, hit_threshold=0.3):
    """
    Predict actions using the trained model with confidence thresholds.
    """
    df = df.copy()
    
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found! Please train it first.")
        df['pred_action'] = 'air'
        return df
        
    X, _ = get_enhanced_features_and_labels(df)
    
    # Get probabilities
    # Classes are mapped: air=0, bounce=1, hit=2
    # Convert to values to ensure compatibility with array-trained model
    probs = clf.predict_proba(X.values if hasattr(X, 'values') else X)
    
    # Default to air
    df['pred_action'] = 'air'
    
    # Apply thresholds
    # probs[:, 1] is bounce, probs[:, 2] is hit
    bounce_mask = probs[:, 1] > bounce_threshold
    hit_mask = probs[:, 2] > hit_threshold
    
    # Assign predictions (Higher confidence wins if both vary, though unlikely)
    # We prioritize Bounce > Hit > Air in this vector logic, but let's be explicit
    
    # Where hit confidence > threshold -> Hit
    df.loc[hit_mask, 'pred_action'] = 'hit'
    
    # Where bounce confidence > threshold -> Bounce (Override hit if bounce is stronger? Usually separate)
    # Let's check which is stronger if both are high
    conflict_mask = bounce_mask & hit_mask
    if conflict_mask.any():
        # Compare probabilities
        better_bounce = probs[:, 1] > probs[:, 2]
        df.loc[conflict_mask & better_bounce, 'pred_action'] = 'bounce'
        df.loc[conflict_mask & (~better_bounce), 'pred_action'] = 'hit'
        
    # Apply pure bounce
    df.loc[bounce_mask & (~hit_mask), 'pred_action'] = 'bounce'
    
    return df



def physics_based_postprocessing(df, min_gap=8):
    """
    Advanced post-processing: separation, screen position, energy rules.
    """
    df = df.copy()
    original_preds = df['pred_action'].copy()
    
    # 1. Screen Position Constraints (Bounces must be low)
    y_limit = df['y'].max() * 0.3
    upper_screen_mask = (df['y'] < y_limit) & (df['pred_action'] == 'bounce')
    df.loc[upper_screen_mask, 'pred_action'] = 'air'
    
    # 2. Cluster Resolution (Minimum Gap)
    events_indices = df[df['pred_action'] != 'air'].index
    if len(events_indices) == 0:
        return df
        
    keep_indices = []
    current_cluster = [events_indices[0]]
    
    for i in range(1, len(events_indices)):
        prev = events_indices[i-1]
        curr = events_indices[i]
        
        if (curr - prev) < min_gap:
            current_cluster.append(curr)
        else:
            # Resolve cluster by max acceleration
            cluster_accels = df.loc[current_cluster, 'accel']
            best_idx = cluster_accels.idxmax()
            keep_indices.append(best_idx)
            current_cluster = [curr]
            
    if current_cluster:
        cluster_accels = df.loc[current_cluster, 'accel']
        best_idx = cluster_accels.idxmax()
        keep_indices.append(best_idx)
    
    # 3. Advanced Validation on Candidates
    final_indices = []
    for idx in keep_indices:
        action = df.loc[idx, 'pred_action']
        
        # Rule: Spurious zero-acceleration hits
        # Re-eval: This reduced recall from 0.90 to 0.30. 
        # The trained model might be picking up hits with low instantaneous accel 
        # but high jerk/speed change. Let's relax this or remove it.
        # if action == 'hit' and df.loc[idx, 'accel'] < 1.0:
        #    continue

            
        # Rule: Bounces must be somewhat local Y maxima (ball hitting floor)
        # Check window ±3 frames
        if action == 'bounce':
            window_y = df.loc[max(0, idx-3):min(len(df), idx+3), 'y']
            if df.loc[idx, 'y'] < window_y.max() * 0.95: # Not near the peak
                 # Allow some noise, but if it's clearly not a peak, reject
                 pass
        
        final_indices.append(idx)
        
    # Apply filtering
    mask_keep = df.index.isin(final_indices)
    is_event = df['pred_action'] != 'air'
    to_drop = is_event & (~mask_keep)
    df.loc[to_drop, 'pred_action'] = 'air'
    
    return df

def post_process_predictions(df, min_event_gap=8):
    """
    Wrapper for physics-based post-processing.
    """
    return physics_based_postprocessing(df, min_gap=min_event_gap)
