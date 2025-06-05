#!/usr/bin/env python3
"""
Database setup script for Healthcare Fraud Detection
Run this to initialize and seed your database
"""

import os
import subprocess
import time
from database_manager import seed_database, DatabaseManager
from config import config

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop.")
        return False

def setup_database():
    """Complete database setup workflow"""
    
    print("🚀 HEALTHCARE FRAUD DETECTION - DATABASE SETUP")
    print("=" * 60)
    
    # Check Docker
    if not check_docker():
        print("❌ Docker is not running. Please start Docker Desktop.")
        return False
    
    # Step 1: Start PostgreSQL using Docker Compose
    print("📦 Starting PostgreSQL database...")
    try:
        subprocess.run(["docker-compose", "-f", "docker-compose-postgres.yml", "up", "-d"], 
                      check=True, capture_output=True)
        print("✅ PostgreSQL started successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start PostgreSQL: {e}")
        return False
    
    # Step 2: Wait for database to be ready
    print("⏳ Waiting for database to be ready...")
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            db = DatabaseManager()
            if db.health_check():
                print("✅ Database is ready!")
                break
        except:
            pass
        
        retry_count += 1
        print(f"   Retry {retry_count}/{max_retries}...")
        time.sleep(3)
    
    if retry_count >= max_retries:
        print("❌ Database failed to start after 30 seconds")
        return False
    
    # Step 3: Seed database with dummy data
    print("🌱 Seeding database with realistic dummy data...")
    
    model_path = config.MODEL_PATHS[0] # Use MODEL_PATH_FULL_PRIMARY from config
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}")
        print("   Continuing without ML-based fraud predictions...")
        model_path = None
    
    # Seed database
    try:
        seeder = seed_database(num_claims=50, model_path=model_path)
        
        # Quick validation
        print("\n🔍 Validating setup...")
        db = DatabaseManager()
        metrics = db.get_dashboard_metrics()
        
        if metrics and metrics.get('total_claims', 0) > 0:
            print(f"✅ Database has {metrics['total_claims']} claims")
            print(f"✅ Risk analysis: {metrics.get('critical_cases', 0)} critical, {metrics.get('high_risk_cases', 0)} high risk")
        else:
            print("⚠️ Database seeded but no claims found")
        
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        return False
    
    print("\n🎉 DATABASE SETUP COMPLETE!")
    print("=" * 60)
    print("✅ PostgreSQL running on localhost:5433")
    print("✅ Database seeded with 50 realistic claims")
    print("✅ Fraud predictions generated and stored")
    print("✅ Ready for Streamlit application")
    print("\n🚀 Next step: Run 'streamlit run streamlit_app.py'")
    print("🌐 pgAdmin available at: http://localhost:8080")
    print("   Login: admin@fraud-detection.com / admin123")
    
    return True

if __name__ == "__main__":
    success = setup_database()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        print("💡 Try running: docker-compose -f docker-compose-postgres.yml down -v")
        print("   Then run this script again.")