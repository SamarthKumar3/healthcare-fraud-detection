import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config

def load_and_merge_data():
    inpatient = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['inpatient']))
    outpatient = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['outpatient']))
    beneficiary = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['beneficiary']))
    main = pd.read_csv(os.path.join(config.DATA_RAW_PATH, config.RAW_DATA_FILES['train']))

    ipop = pd.concat([inpatient, outpatient], axis=0)

    # Aggregate inpatient/outpatient by ClaimID
    ipop_grouped = ipop.groupby("ClaimID").agg({
        "InscClaimAmtReimbursed": "sum",
        "DeductibleAmtPaid": "sum",
        "Provider": "first",
        "BeneID": "first",
        "AdmissionDt": "min",  
        "DischargeDt": "max", 
        "ClaimStartDt": "min",
        "ClaimEndDt": "max"
    }).reset_index()

    # Merge with claims
    merged = main.merge(ipop_grouped, on="Provider", how="left")

    # Merge with beneficiary
    merged = merged.merge(beneficiary, on="BeneID", how="left")

    return merged

def create_fraud_features(df):
    """Create better fraud detection features"""
    print("🔍 Creating improved fraud detection features...")
    
    provider_stats = df.groupby('Provider').agg({
        'InscClaimAmtReimbursed': ['count', 'mean', 'std', 'sum'],
    }).round(2)
    
    provider_stats.columns = ['ProviderClaimCount', 'ProviderAvgClaim', 'ProviderStdClaim', 
                             'ProviderTotalClaims']
    df = df.merge(provider_stats, on='Provider', how='left')
    
    # Patient-level aggregations (calculate BEFORE dropping BeneID)
    patient_stats = df.groupby('BeneID').agg({
        'InscClaimAmtReimbursed': ['count', 'sum', 'mean'],
        'Provider': 'nunique'  # Number of different providers
    }).round(2)
    
    patient_stats.columns = ['PatientClaimCount', 'PatientTotalSpend', 
                           'PatientAvgClaim', 'PatientProviderCount']
    df = df.merge(patient_stats, on='BeneID', how='left')
    
    # CREATE CHRONIC CONDITIONS FEATURES FIRST (before using them)
    chronic_cols = [col for col in df.columns if 'ChronicCond_' in col]
    df['TotalChronicConditions'] = df[chronic_cols].sum(axis=1)
    df['HasMultipleConditions'] = (df['TotalChronicConditions'] > 2).astype(int)
    df['NoChronicConditions'] = (df['TotalChronicConditions'] == 0).astype(int)
    df['HasHeartCondition'] = df.get('ChronicCond_Heartfailure', 0)
    df['HasDiabetes'] = df.get('ChronicCond_Diabetes', 0)
    
    # NOW you can use TotalChronicConditions in other features
    # Provider risk indicators
    df['IsHighVolumeProvider'] = (df['ProviderClaimCount'] > df['ProviderClaimCount'].quantile(0.9)).astype(int)
    df['ProviderClaimStdHigh'] = (df['ProviderStdClaim'] > df['ProviderStdClaim'].quantile(0.9)).astype(int)
    
    # Patient behavior patterns
    df['IsFrequentPatient'] = (df['PatientClaimCount'] > df['PatientClaimCount'].quantile(0.95)).astype(int)
    df['PatientProviderShopping'] = (df['PatientProviderCount'] > 3).astype(int)
    
    # Claim vs provider patterns
    df['ClaimVsProviderAvg'] = df['InscClaimAmtReimbursed'] / (df['ProviderAvgClaim'] + 1)
    df['IsHighVsProviderAvg'] = (df['ClaimVsProviderAvg'] > 2.0).astype(int)
    
    # Medical complexity vs cost patterns (NOW this will work)
    df['CostPerCondition'] = df['InscClaimAmtReimbursed'] / (df['TotalChronicConditions'] + 1)
    df['HighCostLowComplexity'] = ((df['TotalChronicConditions'] <= 1) & 
                                  (df['InscClaimAmtReimbursed'] > df['InscClaimAmtReimbursed'].quantile(0.8))).astype(int)
    
    # 1. CLAIM AMOUNT PATTERNS (more sophisticated)
    df['ClaimAmountLog'] = np.log1p(df['InscClaimAmtReimbursed'])
    df['IsHighValueClaim'] = (df['InscClaimAmtReimbursed'] > 
                             df['InscClaimAmtReimbursed'].quantile(0.95)).astype(int)
    df['IsExtremeValueClaim'] = (df['InscClaimAmtReimbursed'] > 
                                df['InscClaimAmtReimbursed'].quantile(0.99)).astype(int)
    
    # 2. DEDUCTIBLE PATTERNS (instead of raw amount)
    df['DeductibleRatio'] = df['DeductibleAmtPaid'] / (df['InscClaimAmtReimbursed'] + 1)
    df['IsZeroDeductible'] = (df['DeductibleAmtPaid'] == 0).astype(int)
    df['IsLowDeductible'] = (df['DeductibleRatio'] < 0.05).astype(int)
    df['IsHighDeductible'] = (df['DeductibleRatio'] > 0.3).astype(int)
    
    # Remove raw DeductibleAmtPaid
    df = df.drop('DeductibleAmtPaid', axis=1, errors='ignore')
    
    # 4. AGE-BASED RISK FACTORS
    df['IsElderly'] = (df['Age'] > 75).astype(int)
    df['IsVeryElderly'] = (df['Age'] > 85).astype(int)
    df['IsYoungHighCost'] = ((df['Age'] < 35) & 
                            (df['InscClaimAmtReimbursed'] > 15000)).astype(int)
    df['AgeRiskGroup'] = pd.cut(df['Age'], 
                               bins=[0, 35, 65, 75, 85, 100], 
                               labels=[1, 2, 3, 4, 5]).astype(float)
    
    # 5. COVERAGE AND REIMBURSEMENT PATTERNS
    df['PartBCoverageRatio'] = df['NoOfMonths_PartBCov'] / 12
    df['FullYearCoverage'] = ((df['NoOfMonths_PartACov'] == 12) & 
                             (df['NoOfMonths_PartBCov'] == 12)).astype(int)
    
    # Annual amounts per month of coverage
    df['IPMonthlyAvg'] = df['IPAnnualReimbursementAmt'] / (df['NoOfMonths_PartACov'] + 1)
    df['OPMonthlyAvg'] = df['OPAnnualReimbursementAmt'] / (df['NoOfMonths_PartBCov'] + 1)
    
    # 6. PHYSICIAN PATTERNS (if available)
    physician_cols = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
    available_physician_cols = [col for col in physician_cols if col in df.columns]
    if available_physician_cols:
        df['HasMultiplePhysicians'] = (df[available_physician_cols].nunique(axis=1) > 1).astype(int)
    
    # 7. DEMOGRAPHIC RISK FACTORS
    if 'Race' in df.columns:
        # Create risk-based encoding instead of one-hot
        race_risk_map = {1: 1, 2: 1, 3: 2, 5: 1, 0: 1}  # Adjust based on your data analysis
        df['RaceRiskGroup'] = df['Race'].map(race_risk_map).fillna(1)
    
    if 'Gender' in df.columns:
        df['IsMale'] = (df['Gender'] == 1).astype(int)
    
    print("✅ Created improved fraud detection features!")
    return df

def clean_data(df):
    """Enhanced cleaning with new features"""
    import pandas as pd
    import numpy as np

    # Convert PotentialFraud to numeric
    df["PotentialFraud"] = df["PotentialFraud"].map({"Yes": 1, "No": 0})

    # Convert DOB to Age FIRST (needed for fraud features)
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["Age"] = config.FEATURE_THRESHOLDS['reference_year'] - df["DOB"].dt.year
    df.drop(columns=["DOB"], inplace=True, errors="ignore")

    # Convert DOD to IsDead FIRST (needed for fraud features)
    if "DOD" in df.columns:
        df["DOD"] = pd.to_datetime(df["DOD"], errors="coerce")
        df["IsDead"] = df["DOD"].notnull().astype(int)
        df.drop(columns=["DOD"], inplace=True)

    # Create time-based features FIRST (needed for fraud features)
    df["AdmissionDt"] = pd.to_datetime(df["AdmissionDt"], errors="coerce")
    df["DischargeDt"] = pd.to_datetime(df["DischargeDt"], errors="coerce")
    df["HospitalStayDays"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days
    df.drop(columns=["AdmissionDt", "DischargeDt"], inplace=True)

    # Claim duration
    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
    df["ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
    df.drop(columns=["ClaimStartDt", "ClaimEndDt"], inplace=True)

    df = create_fraud_features(df)
    
    if 'RenalDiseaseIndicator' in df.columns:
        print(f"   RenalDiseaseIndicator unique values: {df['RenalDiseaseIndicator'].nunique()}")
        # ALWAYS use simple binary encoding to avoid memory issues
        df['HasRenalDisease'] = (df['RenalDiseaseIndicator'] != 'N').astype(int)
        df = df.drop('RenalDiseaseIndicator', axis=1)
        print(f"   Converted RenalDiseaseIndicator to binary HasRenalDisease feature")
        import gc
        gc.collect()
            
    # NOW drop unneeded high-cardinality or identifier columns
    drop_cols = [
        "ClaimID", "BeneID", "Provider",  
        "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
        "ClmAdmitDiagnosisCode", "DiagnosisGroupCode",
        *[f"ClmDiagnosisCode_{i}" for i in range(1, 11)],
        *[f"ClmProcedureCode_{i}" for i in range(1, 7)],
        "State", "County"
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")        

    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"   Processing categorical column: {col} ({df[col].nunique()} unique values)")
        # ALWAYS use label encoding to avoid memory issues
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"   Converted {col} to numeric using label encoding")
        gc.collect()

    leaky_features = ['ProviderClaimCount', 'ProviderStdClaim', 'ProviderAvgClaim', 
                     'PatientClaimCount', 'PatientTotalClaims', 'ClaimVsProviderAvg','DeductibleAmtPaid','HospitalStayDays', 'IsHighFraudRateProvider','ProviderTotalClaims', 
                     'IsHighVolumeProvider','ProviderClaimStdHigh']
    df = df.drop(columns=leaky_features, errors='ignore')

    # Fill missing numeric columns with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

def save_cleaned_data(df, path=None):
    if path is None:
        path = os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['cleaned_data'])
    df.to_csv(path, index=False)
    print(f"Saved enhanced dataset")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

if __name__ == "__main__":
    print("Loading and merging datasets...")
    df = load_and_merge_data()
    print(f"Raw data shape: {df.shape}")
    

    print("\nCleaning and creating fraud detection features...")
    df_cleaned = clean_data(df)
    print(f"Enhanced data shape: {df_cleaned.shape}")

    print(f"\nNew fraud detection features created:")
    fraud_features = [
        'ClaimVsProviderAvg', 'IsNewProvider', 'IsFrequentClaimant',
        'PatientPaymentRatio', 'IsHighValueClaim', 'IsShortStay', 
        'IsLongStay', 'IsElderly'
    ]
    for feature in fraud_features:
        if feature in df_cleaned.columns:
            print(f"   {feature}")

    save_cleaned_data(df_cleaned, path=os.path.join(config.DATA_PROCESSED_PATH, config.PROCESSED_DATA_FILES['enhanced_data']))