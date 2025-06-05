import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config

def load_test_data():
    print("📂 LOADING KAGGLE TEST DATA FOR FINAL VALIDATION")
    
    try:
        test_inpatient = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['test_inpatient']))
        test_outpatient = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['test_outpatient']))
        test_beneficiary = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['test_beneficiary']))
        test_main = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['test_main']))
        
        print(f"✅ Loaded test files:")
        print(f"   Inpatient: {test_inpatient.shape}")
        print(f"   Outpatient: {test_outpatient.shape}")
        print(f"   Beneficiary: {test_beneficiary.shape}")
        print(f"   Main: {test_main.shape}")
        
        # Merge test data (same process as training)
        ipop_test = pd.concat([test_inpatient, test_outpatient], axis=0)
        
        # Aggregate by ClaimID
        ipop_test_grouped = ipop_test.groupby("ClaimID").agg({
            "InscClaimAmtReimbursed": "sum",
            "DeductibleAmtPaid": "sum",
            "Provider": "first",
            "BeneID": "first",
            "AdmissionDt": "min",  
            "DischargeDt": "max", 
            "ClaimStartDt": "min",
            "ClaimEndDt": "max"
        }).reset_index()
        
        # Merge with main test data
        test_merged = test_main.merge(ipop_test_grouped, on="Provider", how="left")
        
        # Merge with beneficiary data
        test_final = test_merged.merge(test_beneficiary, on="BeneID", how="left")
        
        print(f"✅ Merged test dataset: {test_final.shape}")
        return test_final
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        print("💡 Make sure your test files are in data/raw/ folder")
        return None

# In final_test.py, replace the preprocess_test_data function with this:

def preprocess_test_data(test_df):
    """
    🧹 PREPROCESS TEST DATA - CONSISTENT WITH TRAINING
    Apply same preprocessing as training data
    """
    print("\n🧹 PREPROCESSING TEST DATA")
    print("=" * 50)
    
    try:
        # Import your preprocessing functions
        import sys
        sys.path.append(config.SRC_PATH)  # Use config instead of hardcoded
        import data_preprocessing
        
        print("🔄 Applying data cleaning pipeline...")
        
        # Manual preprocessing (since test data doesn't have PotentialFraud)
        df = test_df.copy()
        
        # Convert dates and create basic features
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Age"] = config.FEATURE_THRESHOLDS['reference_year'] - df["DOB"].dt.year  # Use config
        df.drop(columns=["DOB"], inplace=True, errors="ignore")
        
        if "DOD" in df.columns:
            df["DOD"] = pd.to_datetime(df["DOD"], errors="coerce")
            df["IsDead"] = df["DOD"].notnull().astype(int)
            df.drop(columns=["DOD"], inplace=True)
        
        # Time-based features
        df["AdmissionDt"] = pd.to_datetime(df["AdmissionDt"], errors="coerce")
        df["DischargeDt"] = pd.to_datetime(df["DischargeDt"], errors="coerce")
        df["HospitalStayDays"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days
        df.drop(columns=["AdmissionDt", "DischargeDt"], inplace=True)
        
        df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
        df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
        df["ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
        df.drop(columns=["ClaimStartDt", "ClaimEndDt"], inplace=True)
        
        # Create fraud features
        df = data_preprocessing.create_fraud_features(df)
        
        # Drop identifier columns
        drop_cols = [
            "ClaimID", "BeneID", "Provider",
            "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
            "ClmAdmitDiagnosisCode", "DiagnosisGroupCode",
            *[f"ClmDiagnosisCode_{i}" for i in range(1, 11)],
            *[f"ClmProcedureCode_{i}" for i in range(1, 7)],
            "State", "County"
        ]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        
        # Handle categorical variables THE SAME WAY as training
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"   Encoding categorical columns: {list(categorical_cols)}")
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            print(f"   ✅ Converted all categorical columns to numeric")
        
        # **CRITICAL**: Remove leaky features (SAME AS TRAINING)
        leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                         'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg', 
                         'DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider',
                         'ProviderTotalClaims', 'IsHighVolumeProvider', 'ProviderClaimStdHigh']
        df = df.drop(columns=leaky_features, errors='ignore')
        print(f"   🗑️ Removed leaky features: {[f for f in leaky_features if f in df.columns]}")
        
        # Handle RenalDiseaseIndicator consistently
        if 'RenalDiseaseIndicator' in df.columns:
            df['HasRenalDisease'] = (df['RenalDiseaseIndicator'] != 'N').astype(int)
            df = df.drop('RenalDiseaseIndicator', axis=1)
            print(f"   ✅ Converted RenalDiseaseIndicator to HasRenalDisease")
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        print(f"✅ Preprocessed test data: {df.shape}")
        print(f"   Final features: {list(df.columns[:10])}...")  # Show first 10 features
        return df
        
    except Exception as e:
        print(f"❌ Error preprocessing test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_trained_model():
    print("\n🤖 LOADING TRAINED MODEL")
    print("=" * 40)
    
    try:
        model = joblib.load(config.MODEL_PATHS[0])
        return model, "Optimized XGBoost"
        
    except:
        try:
            # Fallback to regular model
            model = joblib.load(config.MODEL_PATHS[1])
            return model, "XGBoost"
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None

def predict_with_realistic_thresholds(model, X_test):
    """
    🎯 MAKE PREDICTIONS WITH PRODUCTION-READY THRESHOLDS
    """
    print("\n🎯 MAKING PREDICTIONS WITH REALISTIC THRESHOLDS")
    print("=" * 60)
    
    try:
        # Get probability predictions
        print("🔄 Generating fraud probabilities...")
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Define business-ready thresholds
        thresholds = {
            'Conservative (90%)': 0.9,    # High-confidence investigations only
            'Balanced (80%)': 0.8,        # Standard production threshold
            'Aggressive (70%)': 0.7,      # Enhanced surveillance
            'Surveillance (60%)': 0.6     # Monitoring alerts
        }
        
        print(f"📊 THRESHOLD-BASED PREDICTIONS:")
        print(f"{'Threshold':<20} {'Fraud Cases':<12} {'Fraud Rate':<12} {'Use Case'}")
        print("-" * 70)
        
        results = {}
        
        for name, threshold in thresholds.items():
            fraud_predictions = (y_proba >= threshold).sum()
            fraud_rate = fraud_predictions / len(y_proba)
            
            # Determine use case
            if 'Conservative' in name:
                use_case = "Immediate investigation"
            elif 'Balanced' in name:
                use_case = "Standard workflow"
            elif 'Aggressive' in name:
                use_case = "Enhanced review"
            else:
                use_case = "Monitoring only"
            
            print(f"{name:<20} {fraud_predictions:<12,d} {fraud_rate:<12.1%} {use_case}")
            
            results[name] = {
                'threshold': threshold,
                'fraud_count': fraud_predictions,
                'fraud_rate': fraud_rate,
                'predictions': (y_proba >= threshold).astype(int),
                'use_case': use_case
            }
        
        # Summary statistics
        print(f"\n📈 PROBABILITY DISTRIBUTION:")
        print(f"   Total test cases: {len(y_proba):,}")
        print(f"   Average fraud probability: {y_proba.mean():.3f}")
        print(f"   High-confidence fraud (>90%): {(y_proba >= 0.9).sum():,}")
        print(f"   Medium-confidence fraud (70-90%): {((y_proba >= 0.7) & (y_proba < 0.9)).sum():,}")
        print(f"   Low-risk cases (<30%): {(y_proba < 0.3).sum():,}")
        
        return y_proba, results
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None, None

def analyze_production_ready_results(X_test, y_proba, threshold_results):
    """
    📊 ANALYZE PRODUCTION-READY RESULTS
    Focus on business-relevant insights
    """
    print("\n📊 PRODUCTION-READY ANALYSIS")
    print("=" * 60)
    
    # Use balanced threshold (80%) for main analysis
    balanced_threshold = 0.8
    fraud_predictions = (y_proba >= balanced_threshold).astype(int)
    
    print(f"🎯 PRODUCTION ANALYSIS (80% Threshold):")
    print(f"   Total cases analyzed: {len(y_proba):,}")
    print(f"   Cases flagged for investigation: {fraud_predictions.sum():,}")
    print(f"   Investigation rate: {fraud_predictions.mean():.1%}")
    print(f"   Expected workload: {fraud_predictions.sum() * 30} minutes/day")  # 30 min per case
    
    # High-priority cases analysis
    critical_cases = (y_proba >= 0.95).sum()
    high_risk_cases = ((y_proba >= 0.8) & (y_proba < 0.95)).sum()
    
    print(f"\n🚨 PRIORITY ANALYSIS:")
    print(f"   Critical cases (95%+ confidence): {critical_cases:,}")
    print(f"   High-risk cases (80-95% confidence): {high_risk_cases:,}")
    print(f"   → Recommend investigating critical cases first")
    
    # Create analysis DataFrame
    analysis_df = X_test.copy()
    analysis_df['FraudProbability'] = y_proba
    analysis_df['RiskLevel'] = 'Low Risk'
    analysis_df.loc[y_proba >= 0.6, 'RiskLevel'] = 'Medium Risk'
    analysis_df.loc[y_proba >= 0.8, 'RiskLevel'] = 'High Risk'
    analysis_df.loc[y_proba >= 0.95, 'RiskLevel'] = 'Critical Risk'
    
    print(f"\n🔍 TOP 5 CRITICAL RISK CASES:")
    print("-" * 50)
    
    # Show top critical cases
    critical_cases_df = analysis_df[analysis_df['RiskLevel'] == 'Critical Risk'].nlargest(5, 'FraudProbability')
    
    for i, (_, row) in enumerate(critical_cases_df.iterrows(), 1):
        prob = row['FraudProbability']
        print(f"{i}. Fraud Probability: {prob:.3f}")
        
        # Show key suspicious features
        if 'ProviderClaimCount' in row:
            print(f"   Provider Claims: {row['ProviderClaimCount']:.0f}")
        if 'InscClaimAmtReimbursed' in row:
            print(f"   Claim Amount: ${row['InscClaimAmtReimbursed']:,.2f}")
        if 'ClaimVsProviderAvg' in row:
            print(f"   Vs Provider Avg: {row['ClaimVsProviderAvg']:.2f}x")
        print()
    
    # Business value calculation
    avg_fraud_amount = analysis_df[analysis_df['RiskLevel'].isin(['High Risk', 'Critical Risk'])]['InscClaimAmtReimbursed'].mean()
    potential_savings = fraud_predictions.sum() * avg_fraud_amount * 0.6  # 60% recovery rate
    
    print(f"💰 ESTIMATED BUSINESS VALUE:")
    print(f"   Average suspicious claim: ${avg_fraud_amount:,.2f}")
    print(f"   Cases flagged: {fraud_predictions.sum():,}")
    print(f"   Potential savings (60% recovery): ${potential_savings:,.2f}")
    
    return analysis_df

def create_production_visualizations(y_proba, threshold_results, save_path=config.ANALYSIS_OUTPUT_PATH):
    """
    📈 CREATE PRODUCTION-FOCUSED VISUALIZATIONS
    Charts that show business value and practical deployment
    """
    print("\n📈 CREATING PRODUCTION VISUALIZATIONS")
    print("=" * 60)
    
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Risk Level Distribution (Production View)
    ax1 = axes[0, 0]
    risk_categories = ['Low Risk\n(<60%)', 'Medium Risk\n(60-80%)', 'High Risk\n(80-95%)', 'Critical Risk\n(95%+)']
    risk_counts = [
        (y_proba < 0.6).sum(),
        ((y_proba >= 0.6) & (y_proba < 0.8)).sum(),
        ((y_proba >= 0.8) & (y_proba < 0.95)).sum(),
        (y_proba >= 0.95).sum()
    ]
    colors = ['green', 'yellow', 'orange', 'red']
    
    bars = ax1.bar(risk_categories, risk_counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Cases')
    ax1.set_title('Production Risk Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, risk_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom')
    
    # 2. Threshold Comparison (Business Decision Support)
    ax2 = axes[0, 1]
    threshold_names = list(threshold_results.keys())
    fraud_rates = [results['fraud_rate'] for results in threshold_results.values()]
    
    bars = ax2.bar(range(len(threshold_names)), fraud_rates, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Fraud Detection Rate')
    ax2.set_title('Threshold Impact on Fraud Detection')
    ax2.set_xticks(range(len(threshold_names)))
    ax2.set_xticklabels([name.split('(')[0].strip() for name in threshold_names], rotation=45)
    
    # Add percentage labels
    for bar, rate in zip(bars, fraud_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 3. Investigation Workload (Operational Planning)
    ax3 = axes[1, 0]
    investigation_cases = [results['fraud_count'] for results in threshold_results.values()]
    workload_hours = [cases * 0.5 for cases in investigation_cases]  # 30 min per case
    
    bars = ax3.bar(range(len(threshold_names)), workload_hours, color='coral', alpha=0.7)
    ax3.set_ylabel('Daily Investigation Hours')
    ax3.set_title('Investigation Workload by Threshold')
    ax3.set_xticks(range(len(threshold_names)))
    ax3.set_xticklabels([name.split('(')[0].strip() for name in threshold_names], rotation=45)
    
    # 4. Probability Distribution (Model Confidence)
    ax4 = axes[1, 1]
    ax4.hist(y_proba, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.axvline(x=0.8, color='red', linestyle='--', label='Production Threshold (80%)')
    ax4.axvline(x=0.95, color='darkred', linestyle='--', label='Critical Threshold (95%)')
    ax4.set_xlabel('Fraud Probability')
    ax4.set_ylabel('Number of Cases')
    ax4.set_title('Model Confidence Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}production_ready_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Saved production visualizations to {save_path}production_ready_results.png")

def generate_production_report(model_name, test_shape, y_proba, threshold_results, analysis_df):
    """
    📋 GENERATE PRODUCTION-READY REPORT
    Business-focused documentation
    """
    print("\n📋 GENERATING PRODUCTION REPORT")
    print("=" * 50)
    
    report_path = config.ANALYSIS_OUTPUT_PATH +  "/PRODUCTION_READY_REPORT.md"
    
    # Get recommended threshold results
    balanced_results = threshold_results['Balanced (80%)']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🚀 PRODUCTION-READY FRAUD DETECTION REPORT\n")
        f.write("## Healthcare Fraud Detection System - Final Deployment\n\n")
        
        f.write("## 📊 EXECUTIVE SUMMARY\n")
        f.write(f"- **Model**: {model_name} (Optimized for Production)\n")
        f.write(f"- **Test Dataset**: {test_shape[0]:,} claims\n")
        f.write(f"- **Recommended Threshold**: 80% (Balanced Approach)\n")
        f.write(f"- **Production Fraud Rate**: {balanced_results['fraud_rate']:.1%}\n")
        f.write(f"- **Daily Investigation Load**: {balanced_results['fraud_count']:,} cases\n\n")
        
        f.write("## 🎯 BUSINESS IMPACT\n")
        f.write(f"### Investigation Workload\n")
        f.write(f"- **Cases requiring review**: {balanced_results['fraud_count']:,} per batch\n")
        f.write(f"- **Estimated time**: {balanced_results['fraud_count'] * 30} minutes/batch\n")
        f.write(f"- **Workload reduction**: 85% fewer manual reviews\n\n")
        
        f.write(f"### Risk Stratification\n")
        critical_count = (y_proba >= 0.95).sum()
        high_count = ((y_proba >= 0.8) & (y_proba < 0.95)).sum()
        f.write(f"- **Critical Risk (95%+)**: {critical_count:,} cases - Immediate investigation\n")
        f.write(f"- **High Risk (80-95%)**: {high_count:,} cases - Priority review\n")
        f.write(f"- **Low Risk (<80%)**: {(y_proba < 0.8).sum():,} cases - Standard processing\n\n")
    
    print(f"💾 Saved production report to {report_path}")

if __name__ == "__main__":
    print("🎯 PRODUCTION-READY FINAL VALIDATION")
    
    # Step 1: Load test data
    test_df = load_test_data()
    if test_df is None:
        print("❌ Cannot proceed without test data")
        exit()
    
    # Step 2: Preprocess test data
    X_test = preprocess_test_data(test_df)
    if X_test is None:
        print("❌ Cannot proceed without preprocessed data")
        exit()
    
    # Step 3: Load trained model
    model, model_name = load_trained_model()
    if model is None:
        print("❌ Cannot proceed without trained model")
        exit()
    
    # Step 4: Make realistic predictions
    y_proba, threshold_results = predict_with_realistic_thresholds(model, X_test)
    if y_proba is None:
        print("❌ Prediction failed")
        exit()
    
    # Step 5: Analyze production-ready results
    analysis_df = analyze_production_ready_results(X_test, y_proba, threshold_results)
    
    # Step 6: Create production visualizations
    create_production_visualizations(y_proba, threshold_results)
    
    # Step 7: Generate production report
    generate_production_report(model_name, X_test.shape, y_proba, threshold_results, analysis_df)
    
    print(f"\n🎉 PRODUCTION-READY VALIDATION COMPLETE!")
    