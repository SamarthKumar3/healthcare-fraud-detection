import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
import joblib


def load_data():
    """Load the enhanced dataset with fraud features"""
    df = pd.read_csv(os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['enhanced_data']))
    print(f"📊 Loaded enhanced dataset: {df.shape}")
    return df

def split_data(df: pd.DataFrame):
    """Split data and handle encoding - CONSISTENT WITH FEATURE_ANALYSIS"""
    X = df.drop("PotentialFraud", axis=1)
    y = df["PotentialFraud"]
    
    # Handle categorical columns THE SAME WAY as feature_analysis.py
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"🔤 Encoding categorical columns: {list(categorical_cols)}")
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        print(f"   ✅ Converted all categorical columns to numeric")
    
    # Remove leaky features (SAME AS feature_analysis.py)
    leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                     'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg', 
                     'DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider',
                     'ProviderTotalClaims', 'IsHighVolumeProvider']
    X = X.drop(columns=leaky_features, errors='ignore')
    
    # Ensure target is numeric
    y = y.map({'Y': 1, 'N': 0}) if y.dtype == 'object' else y
    
    print(f"📊 Class distribution:")
    print(f"   Non-fraud: {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   Fraud: {(y == 1).sum()} ({(y == 1).mean():.1%})")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def create_models():
    """
    🤖 CREATE MULTIPLE MODELS
    Each model has different strengths for detecting fraud
    """
    # Calculate class weight ratio for imbalanced data
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', 
            random_state=42,
            max_iter=1000
        ),
        
        # Your original model, but balanced
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',  
            random_state=42,
            max_depth=10  
        ),
        
        # Powerful gradient boosting
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
            # XGBoost handles imbalance with scale_pos_weight, we'll set this dynamically
        ),
        
        # Another gradient boosting option
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    return models

def apply_smote(X_train, y_train):
    """
    SMOTE: Create synthetic fraud examples
    Think of this as creating "fake but realistic" fraud cases to balance the dataset
    """
    print("🔄 Applying SMOTE to balance classes...")
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Before SMOTE: {len(y_train)} samples")
    print(f"   After SMOTE: {len(y_balanced)} samples")
    print(f"   Fraud cases: {(y_train == 1).sum()} → {(y_balanced == 1).sum()}")
    
    return X_balanced, y_balanced

def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    """
    TRAIN ALL MODELS AND COMPARE THEM
    """
    models = create_models()
    results = {}
    
    print("🚀 Training multiple models...")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Special handling for XGBoost class imbalance
        if name == 'XGBoost':
            # Calculate scale_pos_weight for XGBoost
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # Fraud probability
        
        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_proba)
        
        print(f"📊 {name} Results:")
        print(f"   ROC-AUC: {auc_score:.3f}")
        print(classification_report(y_val, y_pred, target_names=['Non-Fraud', 'Fraud']))
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'auc_score': auc_score
        }
    
    return results

def train_with_smote(X_train, X_val, y_train, y_val):
    """
    🔄 TRAIN MODELS WITH SMOTE-BALANCED DATA
    """
    print("\n" + "="*60)
    print("🔄 TRAINING WITH SMOTE-BALANCED DATA")
    print("="*60)
    
    # Apply SMOTE
    X_balanced, y_balanced = apply_smote(X_train, y_train)
    
    # Train only the best performing models with SMOTE
    smote_models = {
        'Random Forest + SMOTE': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost + SMOTE': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    smote_results = {}
    
    for name, model in smote_models.items():
        print(f"\n🤖 Training {name}...")
        
        model.fit(X_balanced, y_balanced)
        
        # Evaluate on original validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_proba)
        
        print(f"📊 {name} Results:")
        print(f"   ROC-AUC: {auc_score:.3f}")
        print(classification_report(y_val, y_pred, target_names=['Non-Fraud', 'Fraud']))
        
        smote_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'auc_score': auc_score
        }
    
    return smote_results

def find_best_model(all_results):
    """
    🏆 FIND THE BEST MODEL BASED ON AUC SCORE
    """
    best_auc = 0
    best_model_name = ""
    
    for name, results in all_results.items():
        if results['auc_score'] > best_auc:
            best_auc = results['auc_score']
            best_model_name = name
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   ROC-AUC: {best_auc:.3f}")
    
    return best_model_name, all_results[best_model_name]

def save_best_model(model, model_name):
    """Save the best model for deployment"""
    model_path = os.path.join(os.path.dirname(config.MODEL_PATHS[0]), f"{model_name.replace(' ', '_').lower()}_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Saved best model to {model_path}")

if __name__ == "__main__":
    # Load enhanced data
    df = load_data()
    
    # Split data
    print("\n📂 Splitting data...")
    X_train, X_val, y_train, y_val = split_data(df)
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    # Train baseline models
    print("\n🎯 PHASE 1: Baseline Models (Class Weights)")
    baseline_results = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # Train SMOTE models
    print("\n🎯 PHASE 2: SMOTE-Enhanced Models")
    smote_results = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # Combine all results
    all_results = {**baseline_results, **smote_results}
    
    # Find and save best model
    best_name, best_model = find_best_model(all_results)
    save_best_model(best_model['model'], best_name)
    
    