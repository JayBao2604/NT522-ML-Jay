#!/usr/bin/env python3
"""
Simple test script to diagnose backend startup issues
"""

import sys
import os
import pandas as pd

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✓ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"✗ Google Generative AI import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    # Add the parent directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from train_codebert_model import CodeBERTTrainer
        print("✓ CodeBERTTrainer imported successfully")
    except ImportError as e:
        print(f"✗ CodeBERTTrainer import failed: {e}")
        return False
    
    try:
        from adversarial_learning import AdversarialLearning
        print("✓ AdversarialLearning imported successfully")
    except ImportError as e:
        print(f"✗ AdversarialLearning import failed: {e}")
        return False
    
    try:
        import fga_selection
        print("✓ FGA selection module imported successfully")
    except ImportError as e:
        print(f"✗ FGA selection import failed: {e}")
        return False
    
    return True

def test_attack_pool():
    """Test if attack pool can be loaded"""
    print("\nTesting attack pool...")
    
    attack_pool_path = "attack_pool.csv"
    if not os.path.exists(attack_pool_path):
        print(f"✗ Attack pool file not found: {attack_pool_path}")
        return False
    
    try:
        attack_pool_data = pd.read_csv(attack_pool_path)
        print(f"✓ Attack pool loaded with {len(attack_pool_data)} rows")
        print(f"✓ Columns: {list(attack_pool_data.columns)}")
        
        if 'adversarial_code' in attack_pool_data.columns:
            print("✓ 'adversarial_code' column found")
            non_empty = attack_pool_data['adversarial_code'].dropna()
            print(f"✓ {len(non_empty)} non-empty adversarial code snippets")
        else:
            print("✗ 'adversarial_code' column not found")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed to load attack pool: {e}")
        return False

def test_gemini_config():
    """Test Gemini API configuration"""
    print("\nTesting Gemini configuration...")
    
    try:
        import google.generativeai as genai
        
        # Configure with the API key
        GEMINI_API_KEY = "AIzaSyB2HPcy0LPZKiN2TihoICDdOU_23mhqfa8"
        genai.configure(api_key=GEMINI_API_KEY)
        print("✓ Gemini API configured")
        
        # Try to create model instance
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        print("✓ Gemini model instance created")
        return True
    except Exception as e:
        print(f"✗ Gemini configuration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Backend Diagnostic Test ===\n")
    
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_custom_modules()
    all_passed &= test_attack_pool()
    all_passed &= test_gemini_config()
    
    print(f"\n=== Test Results ===")
    if all_passed:
        print("✓ All tests passed! Backend should start successfully.")
        print("\nTo start the backend, run:")
        print("uvicorn main:app --reload --host 0.0.0.0 --port 3002")
    else:
        print("✗ Some tests failed. Please fix the issues above before starting the backend.")
    
    return all_passed

if __name__ == "__main__":
    main() 