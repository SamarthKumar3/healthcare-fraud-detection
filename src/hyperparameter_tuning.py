import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, make_scorer
import joblib
import time
from itertools import product
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from config import config
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    📂 LOAD DATA AND PREPARE FOR OPTIMIZATION
    """
    print("📂 Loading enhanced dataset for hyperparameter optimization...")
    
    df = pd.read_csv(os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['enhanced_data']))
    
    X = df.drop("PotentialFraud", axis=1)
    y = df["PotentialFraud"]
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Encoding: {list(categorical_cols)}")
        X = pd.get_dummies(X, drop_first=True)
    
    # Ensure target is numeric
    y = y.map({'Y': 1, 'N': 0}) if y.dtype == 'object' else y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Further split training into train/validation for tuning
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"📊 Data split for optimization:")
    print(f"   Training: {X_train_tune.shape[0]:,} samples")
    print(f"   Validation: {X_val_tune.shape[0]:,} samples")
    print(f"   Final test: {X_test.shape[0]:,} samples")
    
    return X_train_tune, X_val_tune, X_test, y_train_tune, y_val_tune, y_test, X.columns.tolist()

def baseline_performance(X_train, X_val, y_train, y_val):
    """
    🎯 ESTABLISH BASELINE PERFORMANCE TO BEAT
    """
    print("\n🎯 ESTABLISHING BASELINE PERFORMANCE")
    print("=" * 60)
    
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    baseline_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=fraud_ratio
    )
    
    print("🚀 Training baseline model...")
    baseline_model.fit(X_train, y_train)
    
    # Evaluate baseline
    y_pred_baseline = baseline_model.predict(X_val)
    y_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]
    baseline_auc = roc_auc_score(y_val, y_proba_baseline)
    
    print(f"📊 BASELINE RESULTS:")
    print(f"   ROC-AUC: {baseline_auc:.4f}")
    print(classification_report(y_val, y_pred_baseline, target_names=['Non-Fraud', 'Fraud']))
    
    return baseline_model, baseline_auc

def quick_parameter_search(X_train, X_val, y_train, y_val, baseline_auc):
    """
    ⚡ QUICK SMART SEARCH FOR BETTER PARAMETERS
    Tests key parameters that usually make the biggest difference
    """
    print("\n⚡ PHASE 1: QUICK SMART PARAMETER SEARCH")
    print("=" * 60)
    
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Smart parameter ranges based on fraud detection best practices
    quick_params = {
        'n_estimators': [100, 200, 300],  # More trees often help
        'max_depth': [4, 6, 8, 10],       # Control overfitting
        'learning_rate': [0.05, 0.1, 0.2], # Learning speed
        'subsample': [0.8, 0.9, 1.0],     # Data sampling
        'colsample_bytree': [0.8, 0.9, 1.0] # Feature sampling
    }
    
    print("🔍 Testing parameter combinations...")
    print("   This will test the most impactful parameters first")
    
    best_auc = baseline_auc
    best_params = None
    best_model = None
    
    # Test a smart subset of combinations
    combinations_tested = 0
    start_time = time.time()
    
    for n_est in quick_params['n_estimators']:
        for max_d in quick_params['max_depth']:
            for lr in quick_params['learning_rate']:
                # Skip some combinations to save time
                if combinations_tested > 30:  # Limit to 30 combinations for speed
                    break
                
                try:
                    model = xgb.XGBClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        learning_rate=lr,
                        subsample=0.9,  # Use good default
                        colsample_bytree=0.9,  # Use good default
                        random_state=42,
                        eval_metric='logloss',
                        scale_pos_weight=fraud_ratio
                    )
                    
                    model.fit(X_train, y_train)
                    y_proba = model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_proba)
                    
                    combinations_tested += 1
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'learning_rate': lr,
                            'subsample': 0.9,
                            'colsample_bytree': 0.9
                        }
                        best_model = model
                        print(f"   🎉 NEW BEST! AUC: {auc:.4f} (n_est={n_est}, depth={max_d}, lr={lr})")
                
                except Exception as e:
                    print(f"   ⚠️ Error with params: {e}")
                    continue
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Quick search complete! Tested {combinations_tested} combinations in {elapsed_time:.1f}s")
    
    if best_params:
        improvement = best_auc - baseline_auc
        print(f"🏆 BEST QUICK SEARCH RESULT:")
        print(f"   ROC-AUC: {best_auc:.4f} (+{improvement:.4f} improvement)")
        print(f"   Best parameters: {best_params}")
    else:
        print("   No improvement found in quick search")
    
    return best_model, best_auc, best_params

def advanced_grid_search(X_train, X_val, y_train, y_val, quick_best_params, quick_best_auc):
    """
    🔬 ADVANCED GRID SEARCH AROUND BEST PARAMETERS
    Fine-tune around the best parameters found in quick search
    """
    print("\n🔬 PHASE 2: ADVANCED FINE-TUNING")
    print("=" * 60)
    
    if quick_best_params is None:
        print("   Using baseline parameters for fine-tuning")
        base_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    else:
        base_params = quick_best_params
    
    print(f"   Fine-tuning around: {base_params}")
    
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Create fine-tuning ranges around best parameters
    fine_tune_params = {
        'n_estimators': [max(50, base_params['n_estimators'] - 50), 
                        base_params['n_estimators'], 
                        base_params['n_estimators'] + 50],
        'max_depth': [max(3, base_params['max_depth'] - 1), 
                     base_params['max_depth'], 
                     min(15, base_params['max_depth'] + 1)],
        'learning_rate': [max(0.01, base_params['learning_rate'] - 0.05), 
                         base_params['learning_rate'], 
                         min(0.3, base_params['learning_rate'] + 0.05)],
        'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
        'reg_lambda': [1, 2, 5],     # L2 regularization
    }
    
    print("🔍 Fine-tuning with regularization parameters...")
    
    best_auc = quick_best_auc
    best_params = quick_best_params
    best_model = None
    
    combinations_tested = 0
    start_time = time.time()
    
    # Test regularization combinations with best base parameters
    for reg_alpha in fine_tune_params['reg_alpha']:
        for reg_lambda in fine_tune_params['reg_lambda']:
            try:
                model = xgb.XGBClassifier(
                    n_estimators=base_params['n_estimators'],
                    max_depth=base_params['max_depth'],
                    learning_rate=base_params['learning_rate'],
                    subsample=base_params.get('subsample', 0.9),
                    colsample_bytree=base_params.get('colsample_bytree', 0.9),
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=fraud_ratio
                )
                
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_proba)
                
                combinations_tested += 1
                
                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        **base_params,
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda
                    }
                    best_model = model
                    print(f"   🎉 NEW BEST! AUC: {auc:.4f} (reg_alpha={reg_alpha}, reg_lambda={reg_lambda})")
            
            except Exception as e:
                print(f"   ⚠️ Error with regularization: {e}")
                continue
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Advanced tuning complete! Tested {combinations_tested} combinations in {elapsed_time:.1f}s")
    
    return best_model, best_auc, best_params

def final_validation(best_model, X_test, y_test, baseline_auc, best_auc):
    """
    🏁 FINAL VALIDATION ON HELD-OUT TEST SET
    The ultimate test of your optimized model
    """
    print("\n🏁 FINAL VALIDATION ON HELD-OUT TEST SET")
    print("=" * 60)
    
    # Evaluate on test set
    y_pred_final = best_model.predict(X_test)
    y_proba_final = best_model.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, y_proba_final)
    
    print(f"🎯 FINAL OPTIMIZED MODEL PERFORMANCE:")
    print(f"   ROC-AUC: {final_auc:.4f}")
    print(f"   Baseline AUC: {baseline_auc:.4f}")
    print(f"   Improvement: +{final_auc - baseline_auc:.4f}")
    
    if final_auc > baseline_auc:
        improvement_pct = ((final_auc - baseline_auc) / baseline_auc) * 100
        print(f"   🎉 SUCCESS! {improvement_pct:.2f}% improvement!")
    else:
        print(f"   📊 No improvement, but baseline was already excellent")
    
    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred_final, target_names=['Non-Fraud', 'Fraud']))
    
    return final_auc

def save_optimized_model(best_model, best_params, final_auc):
    """
    💾 SAVE THE OPTIMIZED MODEL AND PARAMETERS
    """
    print("\n💾 SAVING OPTIMIZED MODEL")
    print("=" * 40)
    
    import os
    os.makedirs(os.path.dirname(config.MODEL_PATHS[0]), exist_ok=True)
    
    # Save model
    model_path = config.MODEL_PATHS[0]
    joblib.dump(best_model, model_path)
    print(f"✅ Saved optimized model to {model_path}")
    
    # Save parameters and performance
    results_path = config.ANALYSIS_OUTPUT_PATH + "/hyperparameter_optimization_results.txt"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Final ROC-AUC: {final_auc:.4f}\n\n")
        f.write("Optimized Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nModel saved to: {model_path}\n")
    
    print(f"✅ Saved optimization results to {results_path}")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data()
    
    # Establish baseline
    baseline_model, baseline_auc = baseline_performance(X_train, X_val, y_train, y_val)
    
    # Quick parameter search
    quick_model, quick_auc, quick_params = quick_parameter_search(
        X_train, X_val, y_train, y_val, baseline_auc
    )
    
    # Advanced fine-tuning
    best_model, best_auc, best_params = advanced_grid_search(
        X_train, X_val, y_train, y_val, quick_params, quick_auc
    )
    
    # Use the best model found (could be baseline, quick, or advanced)
    if best_model is None:
        best_model = baseline_model
        best_params = {'baseline': True}
        best_auc = baseline_auc
    
    # Final validation
    final_auc = final_validation(best_model, X_test, y_test, baseline_auc, best_auc)
    
    # Save optimized model
    save_optimized_model(best_model, best_params, final_auc)
    
    print(f"🏆 Best ROC-AUC achieved: {final_auc:.4f}")
    print(f"📈 Improvement over baseline: +{final_auc - baseline_auc:.4f}")
    print(f"🎯 Your fraud detection system is now MAXIMALLY OPTIMIZED!")