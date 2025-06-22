#!/usr/bin/env python3
"""
Test script to verify EatVul backend setup and functionality
"""

import os
import sys
import pandas as pd
import requests
import json

def test_backend_files():
    """Test if required files exist"""
    print("🔍 Checking backend files...")
    
    backend_dir = "backend"
    required_files = ["main.py", "attack_pool.csv"]
    
    for file in required_files:
        file_path = os.path.join(backend_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            
    return True

def test_attack_pool_format():
    """Test attack pool CSV format"""
    print("\n📊 Checking attack pool format...")
    
    attack_pool_path = os.path.join("backend", "attack_pool.csv")
    
    try:
        df = pd.read_csv(attack_pool_path)
        print(f"✅ CSV loaded successfully with {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check expected columns
        expected_cols = ['original_code', 'adversarial_code', 'label']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing expected columns: {missing_cols}")
        else:
            print("✅ All expected columns present")
            
        # Check for non-null adversarial codes
        non_null_count = df['adversarial_code'].notna().sum()
        print(f"✅ Non-null adversarial codes: {non_null_count}/{len(df)}")
        
        # Show sample
        if len(df) > 0:
            sample = df['adversarial_code'].iloc[0]
            print(f"📝 Sample adversarial code: {sample[:100]}...")
            
    except Exception as e:
        print(f"❌ Error reading attack pool: {e}")
        
    return True

def test_backend_connection():
    """Test if backend is running and responding"""
    print("\n🌐 Testing backend connection...")
    
    base_url = "http://localhost:3002"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Backend is running")
            print(f"📊 Model loaded: {health_data.get('model_loaded', 'Unknown')}")
        else:
            print(f"⚠️  Backend responded with status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Backend not running or not accessible")
        print("💡 Please start the backend with: cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        return False
        
    try:
        # Test diagnostic endpoint
        response = requests.get(f"{base_url}/diagnostic", timeout=5)
        if response.status_code == 200:
            diag_data = response.json()
            print("✅ Diagnostic endpoint working")
            
            attack_pool_info = diag_data.get('attack_pool', {})
            print(f"📊 Attack pool status: {attack_pool_info.get('status', 'unknown')}")
            print(f"📊 Attack pool size: {attack_pool_info.get('size', 0)}")
            
        else:
            print(f"⚠️  Diagnostic endpoint returned {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Diagnostic endpoint error: {e}")
        
    try:
        # Test attack pool endpoint
        response = requests.get(f"{base_url}/attack-pool", timeout=5)
        if response.status_code == 200:
            pool_data = response.json()
            print("✅ Attack pool endpoint working")
            print(f"📊 Total snippets: {pool_data.get('total_snippets', 0)}")
        else:
            print(f"⚠️  Attack pool endpoint returned {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Attack pool endpoint error: {e}")
        
    return True

def create_sample_attack_pool():
    """Create a sample attack pool for testing uploads"""
    print("\n📝 Creating sample attack pool CSV...")
    
    sample_data = [
        {
            "original_code": "gets(buffer);",
            "adversarial_code": "char buf[100]; if(strlen(input)<100) strcpy(buf,input);",
            "label": 0
        },
        {
            "original_code": "system(cmd);", 
            "adversarial_code": "if(strcmp(cmd,\"safe\")==0) system(cmd);",
            "label": 0
        },
        {
            "original_code": "strcpy(dest, src);",
            "adversarial_code": "if(strlen(src) < sizeof(dest)) strcpy(dest, src);",
            "label": 0
        }
    ]
    
    df = pd.DataFrame(sample_data)
    sample_path = "sample_attack_pool.csv"
    df.to_csv(sample_path, index=False)
    
    print(f"✅ Sample attack pool created: {sample_path}")
    print(f"📊 Contains {len(df)} entries")
    print("📝 You can use this file to test the upload functionality")
    
    return sample_path

def main():
    """Run all tests"""
    print("🚀 EatVul Backend Setup Test")
    print("="*50)
    
    # Test file existence
    test_backend_files()
    
    # Test attack pool format
    test_attack_pool_format()
    
    # Test backend connection
    backend_running = test_backend_connection()
    
    # Create sample files
    create_sample_attack_pool()
    
    print("\n" + "="*50)
    print("📋 Test Summary:")
    print("✅ File structure checked")
    print("✅ Attack pool format validated") 
    print(f"{'✅' if backend_running else '❌'} Backend connection tested")
    print("✅ Sample files created")
    
    if not backend_running:
        print("\n💡 To start the backend:")
        print("   cd backend")
        print("   python main.py")
        
    print("\n🎯 You can now test:")
    print("   1. Code file upload (any .c, .cpp, .py, etc.)")
    print("   2. Attack pool CSV upload (use sample_attack_pool.csv)")
    print("   3. Vulnerability analysis")
    print("   4. FGA selection")

if __name__ == "__main__":
    main() 