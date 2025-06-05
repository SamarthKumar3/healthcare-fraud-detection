import psycopg2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import random
import json
import os
from config import config

class DatabaseSeeder:
    """
    Seed the database with realistic dummy claims that will showcase the fraud detection system
    """
    
    def __init__(self, db_config=None):
        if db_config is None:
            self.db_config = config.DB_CONFIG
        else:
            self.db_config = db_config
    
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def clear_existing_data(self):
        """Clear existing data to start fresh"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print("🧹 Clearing existing data...")
                    cursor.execute("DELETE FROM investigations")
                    cursor.execute("DELETE FROM fraud_predictions") 
                    cursor.execute("DELETE FROM claims")
                    conn.commit()
                    print("✅ Existing data cleared")
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def create_realistic_claims(self, num_claims=50):
        """Create realistic claims that showcase TOP 10 FEATURES properly"""
        claims = []
        
        # Provider profiles optimized for top features
        providers = {
            'PRV001': {'risk': 'low', 'avg_claim': 2800, 'specialty': 'Family Practice', 'volume': 150},
            'PRV002': {'risk': 'high', 'avg_claim': 25000, 'specialty': 'Surgery', 'volume': 45}, 
            'PRV003': {'risk': 'medium', 'avg_claim': 4500, 'specialty': 'Internal Medicine', 'volume': 80},
            'PRV004': {'risk': 'low', 'avg_claim': 3200, 'specialty': 'Pediatrics', 'volume': 120},
            'PRV005': {'risk': 'high', 'avg_claim': 18500, 'specialty': 'Cardiology', 'volume': 35},
            'PRV006': {'risk': 'medium', 'avg_claim': 6200, 'specialty': 'Orthopedics', 'volume': 90},
            'PRV007': {'risk': 'critical', 'avg_claim': 35000, 'specialty': 'Neurosurgery', 'volume': 25},
            'PRV008': {'risk': 'suspicious', 'avg_claim': 8500, 'specialty': 'Pain Management', 'volume': 200}
        }
        
        # Patient profiles designed for TOP 10 FEATURES
        patients = []
        for i in range(30):
            age = random.randint(25, 85)
            
            # Create patient patterns that trigger top features
            if i < 3:  # Extremely frequent patients (IsFrequentPatient)
                total_claims = random.randint(50, 100)  
                provider_count = random.randint(8, 15)  # PatientProviderCount
                total_spend = random.randint(200000, 500000)  # PatientTotalSpend
                risk_profile = 'critical'
            elif i < 8:  # High frequency patients
                total_claims = random.randint(20, 49)
                provider_count = random.randint(4, 7)
                total_spend = random.randint(100000, 200000)
                risk_profile = 'high'
            elif i < 15:  # Moderate patients
                total_claims = random.randint(8, 19)
                provider_count = random.randint(2, 4)
                total_spend = random.randint(50000, 100000)
                risk_profile = 'medium'
            else:  # Normal patients
                total_claims = random.randint(1, 7)
                provider_count = random.randint(1, 3)
                total_spend = random.randint(5000, 50000)
                risk_profile = 'low'
            
            # RaceRiskGroup (feature #5)
            race_risk = random.choices([1, 2, 3], weights=[70, 20, 10])[0]  # Higher risk = rarer
            
            patients.append({
                'patient_id': f'PAT{i+1:03d}',
                'age': age,
                'total_claims': total_claims,
                'provider_count': provider_count,
                'total_spend': total_spend,
                'avg_claim': total_spend / max(total_claims, 1),
                'risk_profile': risk_profile,
                'race_risk_group': race_risk
            })
        
        # Generate claims with TOP 10 feature patterns
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_claims):
            provider_id = random.choice(list(providers.keys()))
            provider = providers[provider_id]
            patient = random.choice(patients)
            
            # ClaimAmountLog (TOP FEATURE #1) - Generate amounts that create good log patterns
            base_amount = provider['avg_claim']
            
            # Create claims that will trigger IsExtremeValueClaim (#8)
            if random.random() < 0.1:  # 10% extreme value claims
                claim_amount = random.uniform(75000, 150000)  # Extreme values
            elif patient['risk_profile'] == 'critical':
                claim_amount = base_amount * random.uniform(2.0, 5.0)  # Very high for critical patients
            elif patient['risk_profile'] == 'high':
                claim_amount = base_amount * random.uniform(1.5, 3.0)
            else:
                claim_amount = base_amount * random.uniform(0.5, 2.0)
            
            # Ensure some claims trigger IsHighVsProviderAvg (#2)
            if random.random() < 0.15:  # 15% significantly higher than provider average
                claim_amount = base_amount * random.uniform(3.0, 6.0)
            
            # Hospital stay and duration
            if claim_amount > 20000:
                hospital_days = random.randint(1, 3)  # Short stay high cost
            else:
                hospital_days = random.randint(0, 14)
            
            # Deductible (affects various calculations)
            deductible = min(claim_amount * random.uniform(0.02, 0.15), 2000)
            
            # OPMonthlyAvg (#9) - Create annual outpatient amounts
            op_annual = claim_amount * random.uniform(0.8, 2.0)
            
            # Chronic conditions (affects multiple features)
            chronic_prob = 0.1 + (patient['age'] - 25) / 100  # Age-based probability
            chronic_conditions = []
            if random.random() < chronic_prob:
                conditions = ['diabetes', 'heart_failure', 'cancer', 'kidney']
                chronic_conditions = random.sample(conditions, random.randint(1, 3))
            
            # FullYearCoverage (#10) - Most have full coverage
            full_year_coverage = random.choices([True, False], weights=[85, 15])[0]
            
            # Submission date
            submission_date = base_date + timedelta(
                days=random.randint(0, 30), 
                hours=random.randint(0, 23)
            )
            
            claim = {
                'claim_id': f'CLM{datetime.now().strftime("%Y")}{i+1:04d}',
                'provider_id': provider_id,
                'patient_id': patient['patient_id'],
                'claim_amount': round(claim_amount, 2),
                'deductible_paid': round(deductible, 2),
                'patient_age': patient['age'],
                'hospital_stay_days': hospital_days,
                'claim_duration': max(hospital_days, 1),
                'gender': random.randint(1, 2),
                'chronic_alzheimer': False,
                'chronic_heartfailure': 'heart_failure' in chronic_conditions,
                'chronic_kidney': 'kidney' in chronic_conditions,
                'chronic_cancer': 'cancer' in chronic_conditions,
                'chronic_diabetes': 'diabetes' in chronic_conditions,
                'claim_status': 'PENDING',
                'submission_date': submission_date.strftime('%Y-%m-%d %H:%M:%S'),
                
                # NEW COLUMNS FOR TOP 10 FEATURES
                'patient_total_claims': patient['total_claims'],
                'patient_provider_count': patient['provider_count'],
                'race_risk_group': patient['race_risk_group'],
                'op_annual_amount': round(op_annual, 2),
                
                # For analysis
                'provider_risk': provider['risk'],
                'patient_risk': patient['risk_profile'],
                'expected_fraud_level': self._calculate_expected_fraud_level_new(
                    claim_amount, provider['risk'], patient['risk_profile'], 
                    hospital_days, patient['age'], patient['total_claims'], 
                    patient['provider_count']
                )
            }
            
            claims.append(claim)
        
        return claims

    def _calculate_expected_fraud_level_new(self, amount, provider_risk, patient_risk, 
                                        hospital_days, age, total_claims, provider_count):
        """Updated fraud level calculation based on TOP 10 features"""
        score = 0
        
        # ClaimAmountLog equivalent (high amounts)
        if amount > 100000:
            score += 4
        elif amount > 50000:
            score += 3
        elif amount > 20000:
            score += 2
        
        # Provider patterns
        risk_scores = {'critical': 4, 'suspicious': 3, 'high': 2, 'medium': 1, 'low': 0}
        score += risk_scores.get(provider_risk, 0)
        
        # IsFrequentPatient equivalent
        if total_claims > 50:
            score += 4
        elif total_claims > 20:
            score += 2
        
        # PatientProviderCount (provider shopping)
        if provider_count > 8:
            score += 3
        elif provider_count > 5:
            score += 2
        
        # Short stay high cost
        if hospital_days <= 2 and amount > 15000:
            score += 2
        
        # Age-amount mismatch
        if age < 40 and amount > 25000:
            score += 2
        
        # Convert to risk level
        if score >= 10:
            return 'Critical Risk'
        elif score >= 6:
            return 'High Risk'
        elif score >= 3:
            return 'Medium Risk'
        else:
            return 'Low Risk'
        
    def simple_preprocess_for_seeding(self, claim_data):
        """
        Simple preprocessing for database seeding (fallback when full preprocessing fails)
        """
        try:
            # Create a minimal feature set that matches model expectations
            features = np.array([[
                claim_data['claim_amount'],
                claim_data['deductible_paid'],
                claim_data['gender'],
                1,  # Race
                12, 12,  # Coverage months
                0, claim_data['ChronicCond_Heartfailure'], claim_data['ChronicCond_KidneyDisease'],
                claim_data['ChronicCond_Cancer'], 0, 0, claim_data['ChronicCond_Diabetes'],
                0, 0, 0, 0,  # Other chronic conditions
                claim_data['claim_amount'] * 2,  # IP Annual
                claim_data['deductible_paid'] * 2,  # IP Deductible
                claim_data['claim_amount'],  # OP Annual
                claim_data['deductible_paid'],  # OP Deductible
                claim_data['patient_age'],
                0,  # IsDead
                claim_data['hospital_days'],
                max(claim_data['hospital_days'], 1),  # Claim duration
                5000, 2000, 50,  # Provider stats
                claim_data['claim_amount'] / 5000,  # Claim vs provider avg
                0,  # IsNewProvider
                claim_data['patient_claims'] * 3000,  # Patient total
                claim_data['patient_claims'],
                1 if claim_data['patient_claims'] > 10 else 0,  # Frequent claimant
                claim_data['deductible_paid'] / max(claim_data['claim_amount'], 1),
                1 if claim_data['claim_amount'] > 10000 else 0,  # High value
                1 if claim_data['hospital_days'] <= 1 and claim_data['claim_amount'] > 3000 else 0,  # Short stay
                1 if claim_data['hospital_days'] > 14 else 0,  # Long stay
                1 if claim_data['patient_age'] >= 70 else 0,  # Elderly
                0,  # Death related
                claim_data['ChronicCond_KidneyDisease']  # Renal indicator
            ]])
            
            return features
        except:
            return None

    def calculate_heuristic_fraud_probability(self, claim_data):
        """
        Calculate fraud probability using business rules (fallback when model fails)
        """
        score = 0.1  # Base probability
        
        # High amount
        if claim_data['claim_amount'] > 50000:
            score += 0.4
        elif claim_data['claim_amount'] > 20000:
            score += 0.2
        elif claim_data['claim_amount'] > 10000:
            score += 0.1
        
        # Short stay high cost
        if claim_data['hospital_days'] <= 2 and claim_data['claim_amount'] > 15000:
            score += 0.3
        
        # Frequent claimant
        if claim_data['patient_claims'] > 20:
            score += 0.2
        elif claim_data['patient_claims'] > 10:
            score += 0.1
        
        # Age-amount mismatch
        if claim_data['patient_age'] < 40 and claim_data['claim_amount'] > 25000:
            score += 0.2
        
        # Multiple chronic conditions
        chronic_count = sum([
            claim_data['ChronicCond_Diabetes'],
            claim_data['ChronicCond_Heartfailure'],
            claim_data['ChronicCond_Cancer'],
            claim_data['ChronicCond_KidneyDisease']
        ])
        if chronic_count > 2:
            score += 0.1
        
        return min(score, 0.99) 
    
    def insert_claims(self, claims):
        """Insert claims with new columns for top 10 features"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    print(f"💾 Inserting {len(claims)} claims with top 10 feature support...")
                    
                    for claim in claims:
                        cursor.execute("""
                            INSERT INTO claims (
                                claim_id, provider_id, patient_id, claim_amount, 
                                deductible_paid, patient_age, hospital_stay_days,
                                claim_duration, gender, chronic_alzheimer,
                                chronic_heartfailure, chronic_kidney, chronic_cancer,
                                chronic_diabetes, claim_status, submission_date,
                                patient_total_claims, patient_provider_count, 
                                race_risk_group, op_annual_amount
                            ) VALUES (
                                %(claim_id)s, %(provider_id)s, %(patient_id)s, %(claim_amount)s,
                                %(deductible_paid)s, %(patient_age)s, %(hospital_stay_days)s,
                                %(claim_duration)s, %(gender)s, %(chronic_alzheimer)s,
                                %(chronic_heartfailure)s, %(chronic_kidney)s, %(chronic_cancer)s,
                                %(chronic_diabetes)s, %(claim_status)s, %(submission_date)s,
                                %(patient_total_claims)s, %(patient_provider_count)s,
                                %(race_risk_group)s, %(op_annual_amount)s
                            )
                        """, claim)
                    
                    conn.commit()
                    print("✅ Claims with top 10 feature data inserted successfully")
                    
        except Exception as e:
            print(f"Error inserting claims: {e}")
    
    def run_fraud_predictions_on_dummy_data(self, model_path):
        """
        Run fraud predictions on the dummy data using the trained model
        """
        try:
            # Load the trained model
            model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
            
            # Get claims from database
            with self.get_connection() as conn:
                claims_df = pd.read_sql("""
                    SELECT * FROM claims ORDER BY submission_date DESC
                """, conn)
            
            print(f"🔄 Running predictions on {len(claims_df)} claims...")
            
            predictions_data = []
            
            for _, claim_row in claims_df.iterrows():
                # Convert claim to format expected by preprocessing
                claim_data = {
                    'claim_amount': float(claim_row['claim_amount']),
                    'deductible_paid': float(claim_row['deductible_paid']),
                    'provider_id': claim_row['provider_id'],
                    'patient_claims': random.randint(1, 30),  # Simulated patient history
                    'patient_age': int(claim_row['patient_age']),
                    'hospital_days': int(claim_row['hospital_stay_days']),
                    'gender': int(claim_row['gender']),
                    'ChronicCond_Diabetes': claim_row['chronic_diabetes'],
                    'ChronicCond_Heartfailure': claim_row['chronic_heartfailure'],
                    'ChronicCond_Cancer': claim_row['chronic_cancer'],
                    'ChronicCond_KidneyDisease': claim_row['chronic_kidney'],
                }
                
                # Use simplified preprocessing for seeding
                processed_claim = self.simple_preprocess_for_seeding(claim_data)
                if processed_claim is not None:
                    try:
                        fraud_prob = model.predict_proba(processed_claim)[0, 1]
                    except:
                        # Fallback: create realistic fraud probability based on heuristics
                        fraud_prob = self.calculate_heuristic_fraud_probability(claim_data)
                    
                    # Determine risk level
                    if fraud_prob >= 0.95:
                        risk_level = "Critical Risk"
                    elif fraud_prob >= 0.8:
                        risk_level = "High Risk"
                    elif fraud_prob >= 0.6:
                        risk_level = "Medium Risk"
                    else:
                        risk_level = "Low Risk"
                    
                    # Risk factors
                    risk_factors = {
                        'high_claim_amount': claim_data['claim_amount'] > 15000,
                        'elderly_patient': claim_data['patient_age'] > 70,
                        'short_stay_high_cost': claim_data['hospital_days'] <= 2 and claim_data['claim_amount'] > 10000,
                        'frequent_claimant': claim_data['patient_claims'] > 15,
                        'multiple_chronic_conditions': sum([
                            claim_data['ChronicCond_Diabetes'],
                            claim_data['ChronicCond_Heartfailure'], 
                            claim_data['ChronicCond_Cancer'],
                            claim_data['ChronicCond_KidneyDisease']
                        ]) > 1
                    }
                    
                    predictions_data.append({
                        'claim_id': claim_row['claim_id'],
                        'fraud_probability': float(fraud_prob),
                        'risk_level': risk_level,
                        'risk_score': int(fraud_prob * 100),
                        'risk_factors': json.dumps(risk_factors)
                    })
            
            # Insert predictions into database
            self.insert_predictions(predictions_data)
            
            # Create investigations for high-risk cases
            self.create_investigations_for_high_risk()
            
            print("✅ Fraud predictions completed and stored")
            
        except Exception as e:
            print(f"Error running fraud predictions: {e}")
            

    def get_investigation_queue(self, limit=20):
        """Get investigation queue"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql("""
                    SELECT 
                        i.investigation_id,
                        i.claim_id,
                        i.priority_level,
                        i.status,
                        i.assigned_investigator,
                        i.created_date,
                        c.claim_amount,
                        fp.fraud_probability,
                        fp.risk_level
                    FROM investigations i
                    JOIN claims c ON i.claim_id = c.claim_id
                    LEFT JOIN fraud_predictions fp ON i.prediction_id = fp.prediction_id
                    WHERE i.status IN ('OPEN', 'IN_PROGRESS')
                    ORDER BY 
                        CASE i.priority_level 
                            WHEN 'CRITICAL' THEN 1 
                            WHEN 'HIGH' THEN 2 
                            WHEN 'MEDIUM' THEN 3 
                            ELSE 4 
                        END,
                        i.created_date DESC
                    LIMIT %s
                """, conn, params=[limit])
                return df
        except Exception as e:
            print(f"Error getting investigation queue: {e}")
            return pd.DataFrame()

    def get_setting(self, setting_name):
        """Get system setting value"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT setting_value, setting_type 
                        FROM system_settings 
                        WHERE setting_name = %s
                    """, (setting_name,))
                    result = cursor.fetchone()
                    
                    if result:
                        value, setting_type = result
                        # Convert based on type
                        if setting_type == 'DECIMAL':
                            return float(value)
                        elif setting_type == 'INTEGER':
                            return int(value)
                        elif setting_type == 'BOOLEAN':
                            return value.lower() in ('true', '1', 'yes')
                        else:
                            return value
                    return None
        except Exception as e:
            print(f"Error getting setting {setting_name}: {e}")
            return None
    
    def insert_predictions(self, predictions_data):
        """Insert fraud predictions into database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for pred in predictions_data:
                        cursor.execute("""
                            INSERT INTO fraud_predictions (
                                claim_id, fraud_probability, risk_level, risk_score,
                                risk_factors, model_version
                            ) VALUES (
                                %(claim_id)s, %(fraud_probability)s, %(risk_level)s, 
                                %(risk_score)s, %(risk_factors)s, 'v1.0'
                            )
                        """, pred)
                    conn.commit()
            print(f"✅ Inserted {len(predictions_data)} predictions")
        except Exception as e:
            print(f"Error inserting predictions: {e}")
    
    def create_investigations_for_high_risk(self):
        """Create investigations for high-risk predictions"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create investigations for critical and high risk cases
                    cursor.execute("""
                        INSERT INTO investigations (claim_id, prediction_id, priority_level, status)
                        SELECT 
                            fp.claim_id,
                            fp.prediction_id,
                            CASE 
                                WHEN fp.risk_level = 'Critical Risk' THEN 'CRITICAL'
                                WHEN fp.risk_level = 'High Risk' THEN 'HIGH'
                                ELSE 'MEDIUM'
                            END as priority_level,
                            'OPEN' as status
                        FROM fraud_predictions fp
                        WHERE fp.risk_level IN ('Critical Risk', 'High Risk', 'Medium Risk')
                        AND fp.fraud_probability >= 0.6
                    """)
                    investigations_created = cursor.rowcount
                    conn.commit()
                    
            print(f"✅ Created {investigations_created} investigations for high-risk cases")
        except Exception as e:
            print(f"Error creating investigations: {e}")
    
    def print_summary(self):
        """Print summary of seeded data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get summary statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_claims,
                            AVG(claim_amount) as avg_amount,
                            MAX(claim_amount) as max_amount,
                            MIN(claim_amount) as min_amount
                        FROM claims
                    """)
                    claims_stats = cursor.fetchone()
                    
                    cursor.execute("""
                        SELECT 
                            risk_level,
                            COUNT(*) as count,
                            AVG(fraud_probability) as avg_prob
                        FROM fraud_predictions 
                        GROUP BY risk_level
                        ORDER BY 
                            CASE risk_level 
                                WHEN 'Critical Risk' THEN 1 
                                WHEN 'High Risk' THEN 2 
                                WHEN 'Medium Risk' THEN 3 
                                ELSE 4 
                            END
                    """)
                    risk_stats = cursor.fetchall()
                    
                    cursor.execute("""
                        SELECT COUNT(*) as investigation_count
                        FROM investigations
                        WHERE status = 'OPEN'
                    """)
                    investigation_count = cursor.fetchone()[0]
                    
                    print("\n" + "="*60)
                    print("📊 DATABASE SEEDING SUMMARY")
                    print("="*60)
                    print(f"Total Claims: {claims_stats[0]:,}")
                    print(f"Average Amount: ${claims_stats[1]:,.2f}")
                    print(f"Amount Range: ${claims_stats[3]:,.2f} - ${claims_stats[2]:,.2f}")
                    print(f"Open Investigations: {investigation_count}")
                    
                    print(f"\n🎯 RISK LEVEL DISTRIBUTION:")
                    for risk_level, count, avg_prob in risk_stats:
                        print(f"  {risk_level:<15}: {count:>3} cases ({avg_prob:.1%} avg probability)")
                    
                    print(f"\n✅ Database ready for Streamlit dashboard!")
                    print("="*60)
                    
        except Exception as e:
            print(f"Error getting summary: {e}")

def seed_database(num_claims=50, model_path=None):
    
    seeder = DatabaseSeeder()
    
    seeder.clear_existing_data()
    claims = seeder.create_realistic_claims(num_claims)
    seeder.insert_claims(claims)
    
    if model_path is None:
        for path in config.MODEL_PATHS:
            if os.path.exists(path):
                model_path = path
                break
    
    seeder.print_summary()
    
    return seeder


class DatabaseManager:
    """
    Database manager for fraud detection system
    """
    
    def __init__(self, db_config=None):
        if db_config is None:
            self.db_params = {
                'database': config.DB_CONFIG['database'],
                'user': config.DB_CONFIG['user'],
                'password': config.DB_CONFIG['password'],
                'host': config.DB_CONFIG['host'],
                'port': config.DB_CONFIG['port']
            }
        else:
            self.db_params = db_config
    
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_params)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def health_check(self):
        """Check if database is healthy"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except:
            return False
    
    def get_dashboard_metrics(self):
        """Get dashboard metrics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Total claims
                    cursor.execute("SELECT COUNT(*) FROM claims")
                    total_claims = cursor.fetchone()[0]
                    
                    # Risk level counts
                    cursor.execute("""
                        SELECT 
                            COUNT(CASE WHEN fp.risk_level = 'Critical Risk' THEN 1 END) as critical_cases,
                            COUNT(CASE WHEN fp.risk_level = 'High Risk' THEN 1 END) as high_risk_cases,
                            COUNT(CASE WHEN fp.risk_level = 'Medium Risk' THEN 1 END) as medium_risk_cases
                        FROM fraud_predictions fp
                    """)
                    risk_counts = cursor.fetchone()
                    
                    # Open investigations
                    cursor.execute("SELECT COUNT(*) FROM investigations WHERE status = 'OPEN'")
                    open_investigations = cursor.fetchone()[0]
                    
                    return {
                        'total_claims': total_claims,
                        'critical_cases': risk_counts[0] if risk_counts else 0,
                        'high_risk_cases': risk_counts[1] if risk_counts else 0,
                        'medium_risk_cases': risk_counts[2] if risk_counts else 0,
                        'open_investigations': open_investigations
                    }
        except Exception as e:
            print(f"Error getting dashboard metrics: {e}")
            return {}
    
    def get_claims(self, limit=10):
        """Get recent claims with fraud predictions"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql("""
                    SELECT 
                        c.claim_id,
                        c.claim_amount,
                        c.submission_date,
                        COALESCE(fp.risk_level, 'Not Analyzed') as risk_level,
                        COALESCE(fp.fraud_probability, 0) as fraud_probability
                    FROM claims c
                    LEFT JOIN fraud_predictions fp ON c.claim_id = fp.claim_id
                    ORDER BY c.submission_date DESC
                    LIMIT %s
                """, conn, params=[limit])
                return df
        except Exception as e:
            print(f"Error getting claims: {e}")
            return pd.DataFrame()
    
    def get_predictions_summary(self):
        """Get predictions summary by risk level"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT risk_level, COUNT(*) as count, AVG(fraud_probability) as avg_prob
                        FROM fraud_predictions
                        GROUP BY risk_level
                    """)
                    results = cursor.fetchall()
                    
                    summary = {}
                    for risk_level, count, avg_prob in results:
                        summary[risk_level] = {
                            'count': count,
                            'avg_probability': float(avg_prob) if avg_prob else 0
                        }
                    return summary
        except Exception as e:
            print(f"Error getting predictions summary: {e}")
            return {}
    
    def store_claim(self, claim_data):
        """Store a new claim"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO claims (
                            claim_id, provider_id, patient_id, claim_amount,
                            deductible_paid, patient_age, hospital_stay_days,
                            claim_duration, gender, chronic_alzheimer,
                            chronic_heartfailure, chronic_kidney, chronic_cancer,
                            chronic_diabetes, claim_status, submission_date
                        ) VALUES (
                            %(claim_id)s, %(provider_id)s, %(patient_id)s, %(claim_amount)s,
                            %(deductible_paid)s, %(patient_age)s, %(hospital_stay_days)s,
                            %(claim_duration)s, %(gender)s, %(chronic_alzheimer)s,
                            %(chronic_heartfailure)s, %(chronic_kidney)s, %(chronic_cancer)s,
                            %(chronic_diabetes)s, %(claim_status)s, CURRENT_TIMESTAMP
                        )
                    """, claim_data)
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error storing claim: {e}")
            return False
    
    def store_fraud_prediction(self, claim_id, fraud_probability, risk_level, risk_factors):
        """Store fraud prediction"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO fraud_predictions (
                            claim_id, fraud_probability, risk_level, risk_score, risk_factors
                        ) VALUES (
                            %s, %s, %s, %s, %s
                        ) RETURNING prediction_id
                    """, (
                        claim_id, 
                        fraud_probability, 
                        risk_level, 
                        int(fraud_probability * 100),
                        json.dumps(risk_factors)
                    ))
                    prediction_id = cursor.fetchone()[0]
                    conn.commit()
                    return prediction_id
        except Exception as e:
            print(f"Error storing prediction: {e}")
            return None


def get_database_manager():
    """Get database manager instance"""
    return DatabaseManager()


def init_database():
    """Initialize database connection"""
    return DatabaseManager()


if __name__ == "__main__":
    model_path = config.MODEL_PATHS 
    
    # Seed database with 50 realistic claims
    seeder = seed_database(num_claims=50, model_path=model_path)
    
    print("\n🚀 Database seeding complete!")