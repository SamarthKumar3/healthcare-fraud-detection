import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import joblib
from scipy import stats
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
warnings.filterwarnings('ignore')

def load_model_and_data():
    """
    LOAD YOUR BEST MODEL AND DATA
    """
    df = pd.read_csv(os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['enhanced_data']))
    
    try:
        model = joblib.load(config.MODEL_PATHS[0])
        
        # TEST COMPATIBILITY
        X_test = df.drop("PotentialFraud", axis=1)
        
        # Handle categorical columns the same way
        categorical_cols = X_test.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                X_test[col] = le.fit_transform(X_test[col].astype(str))
        
        # Remove leaky features (same as in prepare_data_for_analysis)
        leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                         'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg', 
                         'DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider',
                         'ProviderTotalClaims', 'IsHighVolumeProvider']
        X_test = X_test.drop(columns=leaky_features, errors='ignore')
        
        # Test if model works with current features
        try:
            _ = model.predict_proba(X_test.iloc[:1])  # Test with 1 row
            print("✅ Loaded compatible model")
            return df, model
        except ValueError as e:
            print(f"❌ Model incompatible: {e}")
            print("🔄 Will retrain with current features...")
            return df, None
            
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("🔄 Will retrain...")
        return df, None

def prepare_data_for_analysis(df):
    """🔧 PREPARE DATA FOR ANALYSIS - Updated for new dataset"""
    print("🔧 Preparing data for feature analysis...")
    
    X = df.drop("PotentialFraud", axis=1)
    y = df["PotentialFraud"]
    
    # REPLACE the categorical encoding section with your memory-efficient approach:
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Found categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            print(f"   {col}: {X[col].nunique()} unique values - using label encoding")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        print(f"   ✅ Converted all categorical columns to numeric")
    else:
        print(f"   No categorical columns found")
    
    # Remove leaky features (same as your cross_validation.py)
    leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                     'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg', 
                     'DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider',
                     'ProviderTotalClaims', 'IsHighVolumeProvider']
    X = X.drop(columns=leaky_features, errors='ignore')
    
    # Ensure target is numeric
    y = y.map({'Y': 1, 'N': 0}) if y.dtype == 'object' else y
    
    print(f"Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X, y, X_train, X_val, y_train, y_val

def quick_retrain_if_needed(model, X_train, y_train):
    """🚀 QUICKLY RETRAIN IF MODEL NOT SAVED - Using your best config"""
    if model is None:
        print("🚀 Quick retraining XGBoost with your optimized parameters...")
        
        # Calculate class weights
        fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Use your current best model config (Balanced model won the batch CV)
        model = xgb.XGBClassifier(
            config.MODEL_PARAMS['optimized']
        )
        
        model.fit(X_train, y_train)
        print("✅ Model retrained successfully with your optimized parameters")
    
    return model

def analyze_feature_importance(model, feature_names, top_n=15):
    """🔍 DEEP DIVE INTO FEATURE IMPORTANCE - Updated for new features"""
    print(f"\n🔍 ANALYZING TOP {top_n} MOST IMPORTANT FEATURES")
    print("=" * 70)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create feature importance DataFrame
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_Pct': (importances / importances.sum()) * 100
    }).sort_values('Importance', ascending=False)
    
    # Display top features
    print(f"🏆 TOP {top_n} FRAUD DETECTION FEATURES:")
    print("-" * 70)
    
    for i, (_, row) in enumerate(feature_df.head(top_n).iterrows(), 1):
        feature_name = row['Feature']
        importance_pct = row['Importance_Pct']
        
        # Updated categorization for your new features
        if any(fraud_feature in feature_name for fraud_feature in 
               ['IsHighVsProviderAvg', 'IsFrequentPatient', 'PatientProviderShopping', 
                'HighCostLowComplexity', 'IsHighValueClaim', 'IsExtremeValueClaim',
                'IsZeroDeductible', 'IsLowDeductible', 'IsHighDeductible']):
            category = "🕵️ YOUR FRAUD FEATURES"
        elif any(financial in feature_name for financial in 
                ['InscClaimAmtReimbursed', 'DeductibleRatio', 'CostPerCondition',
                 'PatientTotalSpend', 'PatientAvgClaim']):
            category = "💰 FINANCIAL"
        elif any(time_feature in feature_name for time_feature in 
                ['ClaimDuration', 'Age', 'IsElderly', 'IsVeryElderly']):
            category = "⏰ TIME/AGE"
        elif any(medical in feature_name for medical in 
                ['ChronicCond_', 'TotalChronicConditions', 'HasMultipleConditions']):
            category = "🏥 MEDICAL"
        else:
            category = "📋 OTHER"
        
        print(f"{i:2d}. {feature_name:<35} {importance_pct:5.1f}% {category}")
    
    # Analyze your custom fraud features performance
    print(f"\n🕵️ YOUR CUSTOM FRAUD FEATURES ANALYSIS:")
    your_fraud_features = [f for f in feature_names if any(fraud_term in f for fraud_term in 
                          ['IsHighVsProviderAvg', 'IsFrequentPatient', 'PatientProviderShopping', 
                           'HighCostLowComplexity', 'IsHighValueClaim', 'IsExtremeValueClaim',
                           'IsZeroDeductible', 'IsLowDeductible', 'IsHighDeductible'])]
    
    fraud_feature_importance = feature_df[feature_df['Feature'].isin(your_fraud_features)]
    total_fraud_importance = fraud_feature_importance['Importance_Pct'].sum()
    
    print(f"   🎯 Your custom fraud features contribute: {total_fraud_importance:.1f}% of model power")
    print(f"   📊 That's {len(fraud_feature_importance)} features out of {len(feature_names)} total")
    
    if total_fraud_importance > 30:
        print("   ✅ EXCELLENT! Your fraud features are driving the model's success!")
    elif total_fraud_importance > 15:
        print("   👍 GOOD! Your fraud features are making a solid contribution")
    else:
        print("   🤔 Your fraud features are helping, but other features dominate")
    
    return feature_df

def analyze_fraud_patterns(df, feature_df, top_features=10):
    """
    🔬 ANALYZE WHAT MAKES FRAUD CASES DIFFERENT
    Understanding the patterns your model learned
    """
    print(f"\n🔬 FRAUD vs NON-FRAUD PATTERN ANALYSIS")
    print("=" * 70)
    
    # Get top features for analysis
    top_feature_names = feature_df.head(top_features)['Feature'].tolist()
    
    # Separate fraud and non-fraud cases
    fraud_cases = df[df['PotentialFraud'] == 1]
    normal_cases = df[df['PotentialFraud'] == 0]
    
    print(f"📊 Analyzing patterns in top {top_features} features:")
    print(f"   Fraud cases: {len(fraud_cases):,}")
    print(f"   Normal cases: {len(normal_cases):,}")
    
    patterns = []
    
    for feature in top_feature_names:
        if feature in df.columns:
            fraud_mean = fraud_cases[feature].mean()
            normal_mean = normal_cases[feature].mean()
            
            # Calculate statistical significance
            if len(fraud_cases[feature].dropna()) > 0 and len(normal_cases[feature].dropna()) > 0:
                stat, p_value = stats.ttest_ind(
                    fraud_cases[feature].dropna(), 
                    normal_cases[feature].dropna()
                )
                
                # Calculate effect size (difference magnitude)
                if normal_mean != 0:
                    ratio = fraud_mean / normal_mean
                else:
                    ratio = fraud_mean if fraud_mean != 0 else 1
                
                patterns.append({
                    'Feature': feature,
                    'Fraud_Avg': fraud_mean,
                    'Normal_Avg': normal_mean,
                    'Ratio': ratio,
                    'P_Value': p_value,
                    'Significant': p_value < 0.001
                })
    
    # Sort by effect size
    patterns_df = pd.DataFrame(patterns)
    patterns_df = patterns_df.sort_values('Ratio', ascending=False)
    
    print(f"\n🚨 KEY FRAUD INDICATORS (What makes claims suspicious):")
    print("-" * 70)
    
    for i, (_, row) in enumerate(patterns_df.head(8).iterrows(), 1):
        feature = row['Feature']
        ratio = row['Ratio']
        fraud_avg = row['Fraud_Avg']
        normal_avg = row['Normal_Avg']
        significant = "***" if row['Significant'] else ""
        
        if ratio > 2:
            direction = f"🔴 {ratio:.1f}x HIGHER"
            interpretation = f"Fraud cases average {fraud_avg:.2f} vs normal {normal_avg:.2f}"
        elif ratio < 0.5:
            direction = f"🔵 {1/ratio:.1f}x LOWER"
            interpretation = f"Fraud cases average {fraud_avg:.2f} vs normal {normal_avg:.2f}"
        else:
            direction = f"⚪ {ratio:.1f}x (similar)"
            interpretation = f"Fraud cases average {fraud_avg:.2f} vs normal {normal_avg:.2f}"
        
        print(f"{i}. {feature:<25} {direction} {significant}")
        print(f"   💡 {interpretation}")
    
    return patterns_df

def create_feature_importance_visualization(feature_df, save_path=config.ANALYSIS_OUTPUT_PATH):
    """
    📊 CREATE VISUAL FEATURE IMPORTANCE CHART
    """
    print(f"\n📊 Creating feature importance visualization...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_15 = feature_df.head(15)
    
    # Create color map for different feature types
    colors = []
    for feature in top_15['Feature']:
        if any(fraud_feature in feature for fraud_feature in 
               ['ClaimVsProvider', 'IsNew', 'IsFrequent', 'Payment', 'IsHigh', 'IsShort', 'IsLong', 'IsElderly']):
            colors.append('#ff6b6b')  # Red for fraud features
        elif any(financial in feature for financial in 
                ['InscClaimAmtReimbursed', 'DeductibleAmtPaid']):
            colors.append('#4ecdc4')  # Teal for financial
        elif any(time_feature in feature for time_feature in 
                ['HospitalStayDays', 'ClaimDuration', 'Age']):
            colors.append('#45b7d1')  # Blue for time
        else:
            colors.append('#96ceb4')  # Green for other
    
    bars = plt.barh(range(len(top_15)), top_15['Importance_Pct'], color=colors)
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('Importance (%)')
    plt.title('🔍 Top 15 Features for Fraud Detection')
    plt.gca().invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, top_15['Importance_Pct'])):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='🕵️ Your Fraud Features'),
        Patch(facecolor='#4ecdc4', label='💰 Financial Features'),
        Patch(facecolor='#45b7d1', label='⏰ Time/Age Features'),
        Patch(facecolor='#96ceb4', label='📋 Other Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to {save_path}feature_importance.png")
    plt.show()

def generate_feature_analysis_report(feature_df, patterns_df, save_path=config.ANALYSIS_OUTPUT_PATH):
    """
    📋 GENERATE COMPREHENSIVE FEATURE ANALYSIS REPORT
    """
    print(f"\n📋 Generating comprehensive feature analysis report...")
    
    import os
    os.makedirs(save_path, exist_ok=True)
    
    report_path = f"{save_path}_feature_analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("📊 EXECUTIVE SUMMARY\n")
        f.write(f"Total Features Analyzed: {len(feature_df)}\n")
        f.write(f"Top Feature Importance: {feature_df.iloc[0]['Importance_Pct']:.1f}%\n")
        
        # Fraud feature analysis
        fraud_features = feature_df[feature_df['Feature'].str.contains('ClaimVsProvider|IsNew|IsFrequent|Payment|IsHigh|IsShort|IsLong|IsElderly')]
        fraud_contribution = fraud_features['Importance_Pct'].sum()
        f.write(f"Your Custom Fraud Features Contribution: {fraud_contribution:.1f}%\n\n")
        
        f.write("🏆 TOP 10 MOST IMPORTANT FEATURES\n")
        f.write("-" * 40 + "\n")
        for i, (_, row) in enumerate(feature_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['Feature']:<30} {row['Importance_Pct']:5.1f}%\n")
        
        f.write(f"\n🚨 KEY FRAUD PATTERNS DISCOVERED\n")
        f.write("-" * 40 + "\n")
        for i, (_, row) in enumerate(patterns_df.head(5).iterrows(), 1):
            f.write(f"{i}. {row['Feature']}: {row['Ratio']:.2f}x difference\n")
    
    print(f"💾 Saved comprehensive report to {report_path}")

if __name__ == "__main__":
    import psutil
    available_memory = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Available memory: {available_memory:.1f} GB")
    
    if available_memory < 2.0:
        print("⚠️ Low memory detected - will use sample for analysis")
    
    # Load data and model
    df, model = load_model_and_data()
    
    # Use sample if memory is low
    if available_memory < 2.0 and len(df) > 100000:
        print(f"📊 Using sample of 100,000 rows for analysis")
        df = df.sample(n=100000, random_state=42)
    
    # Prepare data
    X, y, X_train, X_val, y_train, y_val = prepare_data_for_analysis(df)
    
    # Ensure we have a trained model
    model = quick_retrain_if_needed(model, X_train, y_train)
    
    # Verify model performance
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"🎯 Current model ROC-AUC: {auc_score:.3f}")
    
    # Feature importance analysis
    feature_df = analyze_feature_importance(model, X.columns.tolist())
    
    # Fraud pattern analysis
    patterns_df = analyze_fraud_patterns(df, feature_df)
    
    # Create visualizations
    create_feature_importance_visualization(feature_df)
    
    # Generate comprehensive report
    generate_feature_analysis_report(feature_df, patterns_df)
    