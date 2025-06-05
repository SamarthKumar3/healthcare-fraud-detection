import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import joblib

def load_csv(path):
    """Original function - kept for compatibility"""
    return pd.read_csv(path)


def analyze_feature_importance(model, feature_names, top_n=10):
    """
    🔍 WHICH FEATURES ARE MOST IMPORTANT FOR FRAUD DETECTION?
    This tells us what the model thinks are the biggest fraud indicators
    """
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost, etc.)
        importances = model.feature_importances_
        
        # Create a DataFrame for easy sorting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"🔍 TOP {top_n} FRAUD INDICATORS:")
        print("=" * 50)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
        
        return feature_importance_df
    
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        coefficients = abs(model.coef_[0])  # Take absolute values
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        }).sort_values('Importance', ascending=False)
        
        print(f"🔍 TOP {top_n} FRAUD INDICATORS (Logistic Regression):")
        print("=" * 50)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
        
        return feature_importance_df
    
    else:
        print("❌ This model type doesn't support feature importance analysis")
        return None

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    📊 VISUAL CONFUSION MATRIX
    Shows how many fraud cases we caught vs missed
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Add interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 {title} Breakdown:")
    print(f"   ✅ Correctly identified non-fraud: {tn}")
    print(f"   ✅ Correctly identified fraud: {tp}")
    print(f"   ❌ False alarms (non-fraud called fraud): {fp}")
    print(f"   ❌ Missed fraud cases: {fn}")
    print(f"   💡 We caught {tp}/{tp+fn} fraud cases ({tp/(tp+fn):.1%} recall)")
    
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    📈 ROC CURVE - Shows trade-off between catching fraud and false alarms
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Fraud Caught)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_models_performance(results_dict):
    """
    🏆 COMPARE ALL MODELS SIDE BY SIDE
    Makes it easy to see which model performs best
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        y_pred = results['y_pred']
        y_true = results.get('y_true', None)  # Assuming y_true is stored
        
        if y_true is not None:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': results['auc_score'],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Fraud_Caught': tp,
                'Fraud_Missed': fn
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("🏆 MODEL COMPARISON LEADERBOARD")
    print("=" * 80)
    print(comparison_df.to_string(index=False, float_format='{:.3f}'.format))
    
    return comparison_df

def analyze_fraud_patterns(df, target_col='PotentialFraud'):
    """
    🕵️ ANALYZE PATTERNS IN FRAUD VS NON-FRAUD CASES
    Helps understand what makes fraud cases different
    """
    fraud_cases = df[df[target_col] == 1]
    normal_cases = df[df[target_col] == 0]
    
    print("🕵️ FRAUD vs NORMAL PATTERNS ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    patterns = []
    
    for col in numeric_cols[:10]:  # Analyze top 10 numeric columns
        if col in df.columns:
            fraud_mean = fraud_cases[col].mean()
            normal_mean = normal_cases[col].mean()
            
            if normal_mean != 0:
                ratio = fraud_mean / normal_mean
                patterns.append({
                    'Feature': col,
                    'Fraud_Avg': fraud_mean,
                    'Normal_Avg': normal_mean,
                    'Fraud_vs_Normal_Ratio': ratio
                })
    
    patterns_df = pd.DataFrame(patterns)
    patterns_df = patterns_df.sort_values('Fraud_vs_Normal_Ratio', ascending=False)
    
    print("📊 KEY DIFFERENCES (Fraud vs Normal):")
    for _, row in patterns_df.head(8).iterrows():
        ratio = row['Fraud_vs_Normal_Ratio']
        if ratio > 1.5:
            print(f"🔴 {row['Feature']:<25} {ratio:.2f}x HIGHER in fraud cases")
        elif ratio < 0.7:
            print(f"🔵 {row['Feature']:<25} {1/ratio:.2f}x LOWER in fraud cases")
        else:
            print(f"⚪ {row['Feature']:<25} {ratio:.2f}x (similar)")
    
    return patterns_df

def threshold_optimization(y_true, y_proba):
    """
    🎯 FIND THE BEST THRESHOLD FOR FRAUD DETECTION
    By default, models use 0.5 as the cutoff, but we can optimize this
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
    
    # Find best threshold
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    best_precision = precision[best_threshold_idx]
    best_recall = recall[best_threshold_idx]
    
    print("🎯 THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Default threshold (0.5):")
    default_pred = (y_proba >= 0.5).astype(int)
    default_cm = confusion_matrix(y_true, default_pred)
    default_tn, default_fp, default_fn, default_tp = default_cm.ravel()
    default_precision = default_tp / (default_tp + default_fp) if (default_tp + default_fp) > 0 else 0
    default_recall = default_tp / (default_tp + default_fn) if (default_tp + default_fn) > 0 else 0
    print(f"   Precision: {default_precision:.3f}, Recall: {default_recall:.3f}")
    
    print(f"\nOptimal threshold ({best_threshold:.3f}):")
    print(f"   Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")
    
    # Create optimized predictions
    optimized_pred = (y_proba >= best_threshold).astype(int)
    
    return best_threshold, optimized_pred

def create_fraud_report(model, X_val, y_val, feature_names, model_name="Model"):
    """
    📋 COMPREHENSIVE FRAUD DETECTION REPORT
    Everything you need to know about your model's performance
    """
    print("📋 COMPREHENSIVE FRAUD DETECTION REPORT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Basic metrics
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy:.1%})")
    print(f"   Precision: {precision:.3f} ({precision:.1%})")
    print(f"   Recall:    {recall:.3f} ({recall:.1%})")
    print(f"   F1-Score:  {f1:.3f}")
    
    print(f"\n🎯 FRAUD DETECTION EFFECTIVENESS:")
    print(f"   Total fraud cases in validation: {tp + fn}")
    print(f"   Fraud cases caught: {tp}")
    print(f"   Fraud cases missed: {fn}")
    print(f"   False alarms: {fp}")
    
    # Feature importance
    print(f"\n🔍 TOP FRAUD INDICATORS:")
    feature_importance_df = analyze_feature_importance(model, feature_names, top_n=5)
    
    # Threshold optimization
    print(f"\n🎯 THRESHOLD OPTIMIZATION:")
    best_threshold, optimized_pred = threshold_optimization(y_val, y_proba)
    
    return {
        'confusion_matrix': cm,
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
        'feature_importance': feature_importance_df,
        'best_threshold': best_threshold,
        'optimized_predictions': optimized_pred
    }

def save_model_analysis(analysis_results, model_name, save_path="D:/Web Dev/healthcare-fraud-detection/analysis/"):
    """
    💾 SAVE ANALYSIS RESULTS
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Save feature importance
    if analysis_results['feature_importance'] is not None:
        feature_path = f"{save_path}{model_name}_feature_importance.csv"
        analysis_results['feature_importance'].to_csv(feature_path, index=False)
        print(f"💾 Saved feature importance to {feature_path}")
    
    # Save metrics
    metrics_path = f"{save_path}{model_name}_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        for metric, value in analysis_results['metrics'].items():
            f.write(f"{metric.capitalize()}: {value:.3f}\n")
        f.write(f"Best Threshold: {analysis_results['best_threshold']:.3f}\n")
    
    print(f"💾 Saved metrics to {metrics_path}")


def this_summary(best_model_name, best_auc, baseline_recall, new_recall):
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 ROC-AUC Score: {best_auc:.3f}")
    
    if new_recall > baseline_recall:
        improvement = new_recall - baseline_recall
        print(f"📈 Recall Improvement: {baseline_recall:.1%} → {new_recall:.1%} (+{improvement:.1%})")
        print("✅ Successfully improved fraud detection!")
    else:
        print(f"📊 Recall: {new_recall:.1%} (baseline: {baseline_recall:.1%})")
    
    if new_recall >= 0.65:
        print("✅ Minimum target (65% recall) - ACHIEVED!")
    else:
        print(f"⚠️  Minimum target (65% recall) - Need {0.65 - new_recall:.1%} more")
    
    if new_recall >= 0.75:
        print("🌟 Stretch goal (75% recall) - ACHIEVED!")
    

if __name__ == "__main__":
    print("🛠️ Healthcare Fraud Detection - Analysis Utilities")
    print("Available functions:")
    print("  - analyze_feature_importance()")
    print("  - plot_confusion_matrix()")
    print("  - compare_models_performance()")
    print("  - create_fraud_report()")
    print("  - threshold_optimization()")