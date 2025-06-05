"""
Configuration management for Healthcare Fraud Detection
Loads environment variables and provides default values
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import streamlit as st
# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded environment variables from .env file")
else:
    print("⚠️ No .env file found, using environment variables or defaults")

class Config:
    """Configuration class that loads all settings from environment variables"""
    
    @property
    def DB_CONFIG(self):
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            return {
                'host': st.secrets.database.DB_HOST,
                'port': st.secrets.database.DB_PORT,
                'database': st.secrets.database.DB_NAME,
                'user': st.secrets.database.DB_USER,
                'password': st.secrets.database.DB_PASSWORD,
            }
        else:
            # Fallback to environment variables
            return {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5433'),
                'database': os.getenv('DB_NAME', 'frauddb'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
            }
    
    # ===========================================
    # DATABASE CONFIGURATION
    # ===========================================
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5433'),
        'database': os.getenv('DB_NAME', 'frauddb'),
        'user': os.getenv('DB_USER'),        
        'password': os.getenv('DB_PASSWORD'), 
    }
    # ===========================================
    # MODEL CONFIGURATION
    # ===========================================
    MODEL_PATHS = [
        os.getenv('MODEL_PATH_PRIMARY', 'models/xgboost_optimized_week3.pkl'),
        os.getenv('MODEL_PATH_FALLBACK', 'models/xgboost_model.pkl'),
    ]
    
    # ===========================================
    # DATA PATHS
    # ===========================================
    SRC_PATH = os.getenv('SRC_PATH', 'src')      
    DATA_PROCESSED_PATH = os.getenv('DATA_PROCESSED_PATH', 'data/processed') 
    DATA_RAW_PATH = os.getenv('DATA_RAW_PATH', 'data/raw')                   
    ANALYSIS_OUTPUT_PATH = os.getenv('ANALYSIS_OUTPUT_PATH', 'analysis')  
    
    
    RAW_DATA_FILES = {
        'inpatient': os.getenv('RAW_INPATIENT_FILE', 'Train_Inpatientdata.csv'),
        'outpatient': os.getenv('RAW_OUTPATIENT_FILE', 'Train_Outpatientdata.csv'),
        'beneficiary': os.getenv('RAW_BENEFICIARY_FILE', 'Train_Beneficiarydata.csv'),
        'train': os.getenv('RAW_TRAIN_FILE', 'Train.csv'),
        
        'test_inpatient': os.getenv('RAW_TEST_INPATIENT_FILE', 'Test_Inpatientdata.csv'),
        'test_outpatient': os.getenv('RAW_TEST_OUTPATIENT_FILE', 'Test_Outpatientdata.csv'),
        'test_beneficiary': os.getenv('RAW_TEST_BENEFICIARY_FILE', 'Test_Beneficiarydata.csv'),
        'test_main': os.getenv('RAW_TEST_MAIN_FILE', 'Test.csv')
    }
    
    # NEW: Processed data file names
    PROCESSED_DATA_FILES = {
        'enhanced_data': os.getenv('ENHANCED_DATA_FILE', 'week2_enhanced_data.csv'),
        'cleaned_data': os.getenv('CLEANED_DATA_FILE', 'cleaned_fraud_dataset.csv')
    }
    
    # ===========================================
    # FRAUD DETECTION THRESHOLDS
    # ===========================================
    RISK_THRESHOLDS = {
        'critical': float(os.getenv('THRESHOLD_CRITICAL_RISK', '0.95')),
        'high': float(os.getenv('THRESHOLD_HIGH_RISK', '0.80')),
        'medium': float(os.getenv('THRESHOLD_MEDIUM_RISK', '0.60')),
        'low': float(os.getenv('THRESHOLD_LOW_RISK', '0.00'))
    }
    
    BUSINESS_THRESHOLDS = {
        'conservative': float(os.getenv('THRESHOLD_CONSERVATIVE', '0.90')),
        'balanced': float(os.getenv('THRESHOLD_BALANCED', '0.80')),
        'aggressive': float(os.getenv('THRESHOLD_AGGRESSIVE', '0.70')),
        'surveillance': float(os.getenv('THRESHOLD_SURVEILLANCE', '0.60'))
    }
    
    # ===========================================
    # FEATURE ENGINEERING THRESHOLDS
    # ===========================================
    FEATURE_THRESHOLDS = {
        # Provider analysis thresholds
        'new_provider_claim_threshold': int(os.getenv('NEW_PROVIDER_CLAIM_THRESHOLD', '5')),
        'frequent_claimant_percentile': float(os.getenv('FREQUENT_CLAIMANT_PERCENTILE', '0.95')),
        'high_value_claim_percentile': float(os.getenv('HIGH_VALUE_CLAIM_PERCENTILE', '0.9')),
        'short_stay_threshold': int(os.getenv('SHORT_STAY_THRESHOLD', '2')),
        'long_stay_percentile': float(os.getenv('LONG_STAY_PERCENTILE', '0.95')),
        'elderly_age_threshold': int(os.getenv('ELDERLY_AGE_THRESHOLD', '75')),
        
        # Cross-validation thresholds
        'cv_high_variance_threshold': float(os.getenv('CV_HIGH_VARIANCE_THRESHOLD', '0.05')),
        'cv_moderate_variance_threshold': float(os.getenv('CV_MODERATE_VARIANCE_THRESHOLD', '0.02')),
        'overfitting_significant_gap': float(os.getenv('OVERFITTING_SIGNIFICANT_GAP', '0.02')),
        'overfitting_moderate_gap': float(os.getenv('OVERFITTING_MODERATE_GAP', '0.01')),
        'suspiciously_high_auc': float(os.getenv('SUSPICIOUSLY_HIGH_AUC', '0.99')),
        'excellent_auc': float(os.getenv('EXCELLENT_AUC', '0.95')),
        
        # Feature importance thresholds
        'dominant_feature_threshold': float(os.getenv('DOMINANT_FEATURE_THRESHOLD', '0.5')),
        'provider_importance_threshold': float(os.getenv('PROVIDER_IMPORTANCE_THRESHOLD', '0.3')),
        'patient_importance_threshold': float(os.getenv('PATIENT_IMPORTANCE_THRESHOLD', '0.3')),
        
        # Model selection scoring
        'low_overfitting_score': int(os.getenv('LOW_OVERFITTING_SCORE', '3')),
        'moderate_overfitting_score': int(os.getenv('MODERATE_OVERFITTING_SCORE', '2')),
        'stable_performance_score': int(os.getenv('STABLE_PERFORMANCE_SCORE', '3')),
        'moderate_stability_score': int(os.getenv('MODERATE_STABILITY_SCORE', '2')),
        'good_performance_score': int(os.getenv('GOOD_PERFORMANCE_SCORE', '2')),
        'moderate_performance_score': int(os.getenv('MODERATE_PERFORMANCE_SCORE', '1')),
        'excellent_production_score': int(os.getenv('EXCELLENT_PRODUCTION_SCORE', '6')),
        'good_production_score': int(os.getenv('GOOD_PRODUCTION_SCORE', '4')),
        
        # Data processing thresholds
        'reference_year': int(os.getenv('REFERENCE_YEAR', '2023')),
    }
    
    # ===========================================
    # MODEL HYPERPARAMETERS
    # ===========================================
    MODEL_PARAMS = {
        # Optimized model 
        'optimized': {
            'n_estimators': int(os.getenv('OPTIMIZED_N_ESTIMATORS', '200')),    
            'max_depth': int(os.getenv('OPTIMIZED_MAX_DEPTH', '10')),           
            'learning_rate': float(os.getenv('OPTIMIZED_LEARNING_RATE', '0.2')),
            'subsample': float(os.getenv('OPTIMIZED_SUBSAMPLE', '0.9')),         
            'colsample_bytree': float(os.getenv('OPTIMIZED_COLSAMPLE_BYTREE', '0.9')), 
            'reg_alpha': float(os.getenv('OPTIMIZED_REG_ALPHA', '0.5')),         
            'reg_lambda': float(os.getenv('OPTIMIZED_REG_LAMBDA', '2.0')),       
            'min_child_weight': 5,
            'gamma': 2.0,  
            'random_state': int(os.getenv('OPTIMIZED_RANDOM_STATE', '42')),
            'eval_metric': os.getenv('OPTIMIZED_EVAL_METRIC', 'logloss'),
        },
        
        # Conservative model
        'conservative': {
            'n_estimators': int(os.getenv('CONSERVATIVE_N_ESTIMATORS', '50')),
            'max_depth': int(os.getenv('CONSERVATIVE_MAX_DEPTH', '4')),
            'learning_rate': float(os.getenv('CONSERVATIVE_LEARNING_RATE', '0.1')),
            'subsample': float(os.getenv('CONSERVATIVE_SUBSAMPLE', '0.8')),
            'colsample_bytree': float(os.getenv('CONSERVATIVE_COLSAMPLE_BYTREE', '0.8')),
            'reg_alpha': float(os.getenv('CONSERVATIVE_REG_ALPHA', '1.0')),
            'reg_lambda': float(os.getenv('CONSERVATIVE_REG_LAMBDA', '5.0')),
            'min_child_weight': 5,      # ADD this
            'gamma': 2.0, 
            'random_state': int(os.getenv('CONSERVATIVE_RANDOM_STATE', '42')),
            'eval_metric': os.getenv('CONSERVATIVE_EVAL_METRIC', 'logloss'),
        },
        
        # Very conservative model
        'very_conservative': {
            'n_estimators': int(os.getenv('VERY_CONSERVATIVE_N_ESTIMATORS', '30')),
            'max_depth': int(os.getenv('VERY_CONSERVATIVE_MAX_DEPTH', '3')),
            'learning_rate': float(os.getenv('VERY_CONSERVATIVE_LEARNING_RATE', '0.05')),
            'subsample': float(os.getenv('VERY_CONSERVATIVE_SUBSAMPLE', '0.7')),
            'colsample_bytree': float(os.getenv('VERY_CONSERVATIVE_COLSAMPLE_BYTREE', '0.7')),
            'reg_alpha': float(os.getenv('VERY_CONSERVATIVE_REG_ALPHA', '2.0')),
            'reg_lambda': float(os.getenv('VERY_CONSERVATIVE_REG_LAMBDA', '10.0')),
            'random_state': int(os.getenv('VERY_CONSERVATIVE_RANDOM_STATE', '42')),
            'eval_metric': os.getenv('VERY_CONSERVATIVE_EVAL_METRIC', 'logloss'),
        },
        
        # Balanced model
        'balanced': {
            'n_estimators': int(os.getenv('BALANCED_N_ESTIMATORS', '100')),
            'max_depth': int(os.getenv('BALANCED_MAX_DEPTH', '6')),
            'learning_rate': float(os.getenv('BALANCED_LEARNING_RATE', '0.15')),
            'subsample': float(os.getenv('BALANCED_SUBSAMPLE', '0.8')),
            'colsample_bytree': float(os.getenv('BALANCED_COLSAMPLE_BYTREE', '0.8')),
            'reg_alpha': float(os.getenv('BALANCED_REG_ALPHA', '0.5')),
            'reg_lambda': float(os.getenv('BALANCED_REG_LAMBDA', '2.0')),
            'random_state': int(os.getenv('BALANCED_RANDOM_STATE', '42')),
            'eval_metric': os.getenv('BALANCED_EVAL_METRIC', 'logloss'),
        }
    }
    
    # ===========================================
    # CROSS-VALIDATION CONFIGURATION
    # ===========================================
    CV_CONFIG = {
        'n_splits': int(os.getenv('CV_N_SPLITS', '5')),
        'shuffle': os.getenv('CV_SHUFFLE', 'true').lower() == 'true',
        'random_state': int(os.getenv('CV_RANDOM_STATE', '42')),
        'test_size': float(os.getenv('CV_TEST_SIZE', '0.3')),
        'scoring_metric': os.getenv('CV_SCORING_METRIC', 'roc_auc'),
        'n_jobs': int(os.getenv('CV_N_JOBS', '1'))
    }
    
    # ===========================================
    # PROVIDER RISK PROFILES
    # ===========================================
    @staticmethod
    def get_provider_profiles() -> Dict[str, Any]:
        """Get provider risk profiles from environment or use defaults"""
        try:
            profiles_json = os.getenv('PROVIDER_PROFILES')
            if profiles_json:
                return json.loads(profiles_json)
        except json.JSONDecodeError:
            print("⚠️ Error parsing PROVIDER_PROFILES JSON, using defaults")
        
        # Default provider profiles
        return {
            'PRV001': {'risk': 'low', 'base_avg': 2800, 'base_count': 150},
            'PRV002': {'risk': 'high', 'base_avg': 25000, 'base_count': 45},
            'PRV003': {'risk': 'medium', 'base_avg': 4500, 'base_count': 80},
            'PRV004': {'risk': 'low', 'base_avg': 3200, 'base_count': 120},
            'PRV005': {'risk': 'high', 'base_avg': 18500, 'base_count': 35},
            'PRV006': {'risk': 'medium', 'base_avg': 6200, 'base_count': 90},
            'PRV007': {'risk': 'critical', 'base_avg': 35000, 'base_count': 25},
            'PRV008': {'risk': 'suspicious', 'base_avg': 8500, 'base_count': 200}
        }
    
    # ===========================================
    # PATIENT CLAIM THRESHOLDS
    # ===========================================
    PATIENT_THRESHOLDS = {
        'extremely_frequent': int(os.getenv('PATIENT_CLAIMS_EXTREMELY_FREQUENT', '100')),
        'very_frequent': int(os.getenv('PATIENT_CLAIMS_VERY_FREQUENT', '50')),
        'frequent': int(os.getenv('PATIENT_CLAIMS_FREQUENT', '25')),
        'above_average': int(os.getenv('PATIENT_CLAIMS_ABOVE_AVERAGE', '15')),
        'moderate': int(os.getenv('PATIENT_CLAIMS_MODERATE', '10'))
    }
    
    # ===========================================
    # FINANCIAL THRESHOLDS
    # ===========================================
    FINANCIAL_THRESHOLDS = {
        'high_value_claim': float(os.getenv('HIGH_VALUE_CLAIM_THRESHOLD', '15000')),
        'very_high_claim': float(os.getenv('VERY_HIGH_CLAIM_THRESHOLD', '100000')),
        'extreme_high_claim': float(os.getenv('EXTREME_HIGH_CLAIM_THRESHOLD', '50000')),
        'deductible_very_low': float(os.getenv('DEDUCTIBLE_RATIO_VERY_LOW', '0.01')),
        'deductible_low': float(os.getenv('DEDUCTIBLE_RATIO_LOW', '0.03')),
        'deductible_high': float(os.getenv('DEDUCTIBLE_RATIO_HIGH', '0.40')),
        'provider_ratio_critical': float(os.getenv('PROVIDER_CLAIM_RATIO_CRITICAL', '3.0')),
        'provider_ratio_warning': float(os.getenv('PROVIDER_CLAIM_RATIO_WARNING', '1.5')),
        'provider_volume_high': int(os.getenv('PROVIDER_VOLUME_HIGH', '2000')),
        'provider_volume_low': int(os.getenv('PROVIDER_VOLUME_LOW', '50'))
    }
    
    # ===========================================
    # AGE-BASED THRESHOLDS
    # ===========================================
    AGE_THRESHOLDS = {
        'elderly': int(os.getenv('AGE_ELDERLY_THRESHOLD', '70')),
        'young_high_cost': int(os.getenv('AGE_YOUNG_HIGH_COST', '35')),
        'middle_high_cost': int(os.getenv('AGE_MIDDLE_HIGH_COST', '50')),
        'young_cost_threshold': float(os.getenv('YOUNG_PATIENT_HIGH_COST_THRESHOLD', '30000')),
        'middle_cost_threshold': float(os.getenv('MIDDLE_AGE_HIGH_COST_THRESHOLD', '75000'))
    }
    
    # ===========================================
    # MEDICAL THRESHOLDS
    # ===========================================
    MEDICAL_THRESHOLDS = {
        'short_stay_days': int(os.getenv('SHORT_STAY_DAYS', '1')),
        'long_stay_days': int(os.getenv('LONG_STAY_DAYS', '14')),
        'short_stay_high_cost': float(os.getenv('SHORT_STAY_HIGH_COST', '25000')),
        'outpatient_high_cost': float(os.getenv('OUTPATIENT_HIGH_COST', '15000')),
        'no_chronic_high_cost': float(os.getenv('NO_CHRONIC_HIGH_COST', '25000')),
        'multiple_conditions': int(os.getenv('MULTIPLE_CONDITIONS_THRESHOLD', '4'))
    }
    
    # ===========================================
    # STREAMLIT CONFIGURATION
    # ===========================================
    STREAMLIT_CONFIG = {
        'title': os.getenv('APP_TITLE', 'Healthcare Fraud Detection'),
        'icon': os.getenv('APP_ICON', '🏥'),
        'layout': os.getenv('PAGE_LAYOUT', 'wide')
    }
    
    # Default form values
    FORM_DEFAULTS = {
        'claim_id_prefix': os.getenv('DEFAULT_CLAIM_ID_PREFIX', 'CLM_MANUAL_'),
        'provider_id': os.getenv('DEFAULT_PROVIDER_ID', 'PRV001'),
        'patient_id': os.getenv('DEFAULT_PATIENT_ID', 'PAT001'),
        'claim_amount': float(os.getenv('DEFAULT_CLAIM_AMOUNT', '15000.0')),
        'deductible': float(os.getenv('DEFAULT_DEDUCTIBLE', '300.0')),
        'patient_age': int(os.getenv('DEFAULT_PATIENT_AGE', '65')),
        'hospital_days': int(os.getenv('DEFAULT_HOSPITAL_DAYS', '2')),
        'patient_claims': int(os.getenv('DEFAULT_PATIENT_CLAIMS', '5'))
    }
    
    # ===========================================
    # SYSTEM CONFIGURATION
    # ===========================================
    SYSTEM_CONFIG = {
        'coverage_months_a': int(os.getenv('COVERAGE_MONTHS_PART_A', '12')),
        'coverage_months_b': int(os.getenv('COVERAGE_MONTHS_PART_B', '12')),
        'prediction_timeout': int(os.getenv('MODEL_PREDICTION_TIMEOUT', '30')),
        'preprocessing_timeout': int(os.getenv('PREPROCESSING_TIMEOUT', '20'))
    }
    
    
    # ===========================================
    # LOGGING CONFIGURATION
    # ===========================================
    LOGGING_CONFIG = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'file_path': os.getenv('LOG_FILE_PATH', 'logs/fraud_detection.log'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_paths = [cls.SRC_PATH]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        
        if missing_paths:
            print(f"⚠️ Missing required paths: {missing_paths}")
            return False
        
        # Validate model paths
        model_exists = any(os.path.exists(path) for path in cls.MODEL_PATHS)
        if not model_exists:
            print(f"⚠️ No model found at any of these paths: {cls.MODEL_PATHS}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration"""
        print("\n" + "="*50)
        print("📋 CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Database: {cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}")
        print(f"Model Paths: {len(cls.MODEL_PATHS)} configured")
        print(f"Source Path: {cls.SRC_PATH}")
        print(f"Risk Thresholds: {cls.RISK_THRESHOLDS}")
        print(f"Debug Mode: {cls.LOGGING_CONFIG['debug_mode']}")
        print("="*50)

# Create a global config instance
config = Config()

# Validate configuration on import
if __name__ != "__main__":
    config.validate_config()