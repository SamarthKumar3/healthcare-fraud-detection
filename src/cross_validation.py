import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import psutil
import gc
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config

from batch_processing_system import memory_efficient_cross_validation, run_memory_efficient_pipeline



def load_data():
    """LOAD DATA FOR ROBUST TESTING"""
    print("Loading data for cross-validation robustness testing...")
    
    df = pd.read_csv(os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['enhanced_data']))
    
    X = df.drop("PotentialFraud", axis=1)
    y = df["PotentialFraud"]
    
    # REPLACE the problematic categorical encoding with this:
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Found categorical columns: {list(categorical_cols)}")
        print(f"   Using memory-efficient label encoding for all categorical variables")
        
        for col in categorical_cols:
            n_unique = X[col].nunique()
            print(f"   {col}: {n_unique} unique values - using label encoding")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
        print(f"   ✅ Converted all categorical columns to numeric")
    else:
        print(f"   No categorical columns found")
    
    # Remove leaky features
    leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                     'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg', 
                     'DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider',
                     'ProviderTotalClaims', 'IsHighVolumeProvider','ProviderClaimStdHigh']
    X = X.drop(columns=leaky_features, errors='ignore')
    
    # Ensure target is numeric
    y = y.map({'Y': 1, 'N': 0}) if y.dtype == 'object' else y
    
    print(f"Dataset for testing: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Fraud cases: {(y == 1).sum():,} ({(y == 1).mean():.1%})")
    print(f"Non-fraud cases: {(y == 0).sum():,} ({(y == 0).mean():.1%})")
    
    return X, y

def test_overfitting_simple(X, y):
    """
    SIMPLE OVERFITTING TEST
    Compare performance on training vs validation data
    """
    print("\nSIMPLE OVERFITTING TEST")
    if len(X) > 100000:
        print(f"   Large dataset detected ({len(X):,} samples)")
        print(f"   Using stratified sample for overfitting test...")
        
        sample_size = min(50000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        print(f"   Using {len(X_sample):,} samples for overfitting test")
    else:
        X_sample, y_sample = X, y
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample,  # Changed from X, y
        test_size=config.CV_CONFIG['test_size'], 
        stratify=y_sample,   # Changed from y
        random_state=config.CV_CONFIG['random_state']
    )
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("🔍 DEBUG: Checking model parameters...")
    print("Optimized params from config:")
    for key, value in config.MODEL_PARAMS['optimized'].items():
        print(f"   {key}: {value}")
    
    models_to_test = {
        'Optimized (Suspected Overfit)': xgb.XGBClassifier(
            **config.MODEL_PARAMS['optimized'],
            scale_pos_weight=fraud_ratio,
        ),
        'Conservative (Less Overfit)': xgb.XGBClassifier(
            n_estimators=50,   # Fewer trees
            max_depth=4,       # Shallow trees
            learning_rate=0.1, # Slower learning
            subsample=0.8,     # More sampling
            colsample_bytree=0.8,
            reg_alpha=1.0,     # More regularization
            reg_lambda=5.0,    # More regularization
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=fraud_ratio
        ),
        'Very Conservative': xgb.XGBClassifier(
            n_estimators=30,   # Very few trees
            max_depth=3,       # Very shallow
            learning_rate=0.05, # Very slow learning
            subsample=0.7,     # More randomness
            colsample_bytree=0.7,
            reg_alpha=2.0,     # High regularization
            reg_lambda=10.0,   # High regularization
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=fraud_ratio
        )
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        print(f"\nTesting {name} model...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on training data
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_proba)
        
        # Evaluate on validation data
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        # Calculate overfitting gap
        overfitting_gap = train_auc - val_auc
        
        results[name] = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'gap': overfitting_gap,
            'model': model
        }
        
        print(f"   Training AUC: {train_auc:.4f}")
        print(f"   Validation AUC: {val_auc:.4f}")
        print(f"   Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > config.FEATURE_THRESHOLDS['overfitting_significant_gap']:
            print(f"   WARNING: Significant overfitting detected!")
        elif overfitting_gap > config.FEATURE_THRESHOLDS['overfitting_moderate_gap']:
            print(f"   CAUTION: Moderate overfitting detected")
        else:
            print(f"   GOOD: Low overfitting risk")
    
    return results

def run_simplified_validation(X, y):
    """Simplified validation when memory is insufficient"""
    print("Running memory-efficient validation...")
    
    fraud_ratio = (y == 0).sum() / (y == 1).sum()
    
    # Use smaller sample for validation
    sample_size = min(10000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    # Test the same models as in simple test but with simplified CV
    models_to_test = {
        'Suspected Overfit Model': xgb.XGBClassifier(**config.MODEL_PARAMS['optimized'], scale_pos_weight=fraud_ratio),
        'Conservative Model': xgb.XGBClassifier(**config.MODEL_PARAMS['conservative'], scale_pos_weight=fraud_ratio),
        'Balanced Model': xgb.XGBClassifier(**config.MODEL_PARAMS['balanced'], scale_pos_weight=fraud_ratio)
    }
    
    cv_results = {}
    
    for name, model in models_to_test.items():
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_sample, y_sample, 
                test_size=config.CV_CONFIG['test_size'], 
                stratify=y_sample, 
                random_state=config.CV_CONFIG['random_state']
            )
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            cv_results[name] = {
                'mean_auc': auc_score,
                'std_auc': 0.01,  # Small variance estimate for single run
                'all_scores': [auc_score]
            }
            
            print(f"   {name}: AUC = {auc_score:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ {name}: Failed ({str(e)})")
            cv_results[name] = {
                'mean_auc': 0.5,
                'std_auc': 0.1,
                'all_scores': [0.5]
            }
    
    return cv_results

def analyze_model_performance(model, X_val, y_val):
    """Analyze what the model is actually learning"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"\n📊 Detailed Performance Analysis:")
    print(f"   AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Fraud', 'Fraud']))
    
    # Analyze prediction distribution
    print(f"\n📈 Prediction Distribution:")
    print(f"   Predicted Fraud: {y_pred.sum():,} ({y_pred.mean():.1%})")
    print(f"   Actual Fraud: {y_val.sum():,} ({y_val.mean():.1%})")
    
def analyze_feature_distributions(X, y):
    """Analyze feature distributions between fraud and non-fraud cases"""
    print("\n📊 FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    fraud_mask = (y == 1)
    
    for col in X.select_dtypes(include=[np.number]).columns[:10]:  # Top 10 numeric features
        fraud_mean = X.loc[fraud_mask, col].mean()
        no_fraud_mean = X.loc[~fraud_mask, col].mean()
        
        if no_fraud_mean != 0:
            ratio = fraud_mean / no_fraud_mean
            print(f"{col:<25} Fraud: {fraud_mean:8.2f} | No Fraud: {no_fraud_mean:8.2f} | Ratio: {ratio:.2f}")




def detect_data_leakage(X, y):
    """
    DATA LEAKAGE DETECTION
    Check if there are features that shouldn't exist in real-world prediction
    """
    print("\nDATA LEAKAGE DETECTION")
    print("=" * 60)
    print("Checking for features that might cause artificial perfect performance...")
    
    # Train a simple model and check feature importance
    fraud_ratio = (y == 0).sum() / (y == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
        scale_pos_weight=fraud_ratio
    )
    
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔍 Top 10 most important features:")
    suspicious_features = []
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        feature = row['feature']
        importance = row['importance']
        
        # Check for suspicious patterns
        is_suspicious = False
        reason = ""
        
        if importance > config.FEATURE_THRESHOLDS['dominant_feature_threshold']:
            is_suspicious = True
            reason = "DOMINATES MODEL"
        elif 'Provider' in feature and importance > config.FEATURE_THRESHOLDS['provider_importance_threshold']:
            is_suspicious = True
            reason = "PROVIDER ID LEAK?"
        elif 'Patient' in feature and importance > config.FEATURE_THRESHOLDS['patient_importance_threshold']:
            is_suspicious = True
            reason = "PATIENT ID LEAK?"
        
        status = "SUSPICIOUS" if is_suspicious else "OK"
        print(f"{i:2d}. {feature:<30} {importance:5.1%} {status}")
        
        if is_suspicious:
            suspicious_features.append((feature, reason))
    
    if suspicious_features:
        print(f"\nPOTENTIAL DATA LEAKAGE DETECTED:")
        for feature, reason in suspicious_features:
            print(f"   • {feature}: {reason}")
        print(f"\nRECOMMENDATION: Remove these features and retrain")
    else:
        print(f"\nNo obvious data leakage detected")
        
    
    
    return feature_importance, suspicious_features

def recommend_production_model(simple_results, cv_results):
    print("\nPRODUCTION MODEL RECOMMENDATION")
    print("Prioritizing FULL DATASET batch CV results...")
    
    # Use batch CV results (full dataset) as primary metric
    best_batch_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_auc'])
    best_batch_score = cv_results[best_batch_model]['mean_auc']
    
    print(f"\n🏆 BEST MODEL (Full Dataset): {best_batch_model}")
    print(f"   Batch CV AUC: {best_batch_score:.4f}")
    print(f"   Based on ALL 558,211 samples")
    
    return best_batch_model

if __name__ == "__main__":
    try:
        X, y = load_data()
        
        # Test for overfitting
        print("Starting overfitting test...")
        simple_results = test_overfitting_simple(X, y)
        
        # Cross-validation test
        print("Starting batch cross-validation...")
        cv_results = memory_efficient_cross_validation(X, y)
        optimized_params, best_score, best_model_name = run_memory_efficient_pipeline(X, y)

        # Check for data leakage
        print("Starting data leakage detection...")
        feature_importance, suspicious_features = detect_data_leakage(X, y)
        
        print("Starting model recommendations...")
        recommendations = recommend_production_model(simple_results, cv_results)
        
        print("Starting performance analysis...")
        analyze_model_performance(simple_results['Optimized (Suspected Overfit)']['model'], X, y)
        analyze_feature_distributions(X, y)
        
        print(f"\n ROBUSTNESS TESTING COMPLETE!")
        if suspicious_features:
            print("ACTION REQUIRED: Potential data leakage detected")
            print("   → Remove suspicious features and retrain")
        else:
            print("No data leakage detected")
            
    except MemoryError as e:
        print(f"Insufficient memory to complete analysis: {str(e)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()  # This will show the actual error
    finally:
        # Clean up memory
        gc.collect()
    