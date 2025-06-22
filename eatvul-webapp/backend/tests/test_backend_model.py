#!/usr/bin/env python3
"""
Test script to verify the backend model loading works correctly
"""
import os
import sys
import traceback

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_codebert_model import CodeBERTTrainer

def test_backend_model_loading():
    """Test the exact model loading process used by the backend"""
    print("🔍 Testing Backend Model Loading Process")
    print("=" * 60)
    
    # Simulate the exact backend startup process
    print("🔄 Initializing CodeBERTTrainer...")
    try:
        codebert_trainer = CodeBERTTrainer()
        print("✅ CodeBERTTrainer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize CodeBERTTrainer: {e}")
        return False
    
    # Use the exact path resolution from backend
    backend_file_path = os.path.abspath(__file__)
    backend_dir = os.path.dirname(backend_file_path)
    model_path = os.path.join(os.path.dirname(backend_dir), "codebert-model")
    
    print(f"📁 Backend file: {backend_file_path}")
    print(f"📁 Backend directory: {backend_dir}")
    print(f"📁 Model path: {model_path}")
    print(f"📁 Path exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        return False
    
    # Check required files
    required_files = ["best_model.pt", "model_config.json", "special_tokens_map.json"]
    missing_files = []
    
    print(f"📂 Model directory contents:")
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isfile(item_path):
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"   {item} ({size_mb:.1f} MB)")
        else:
            print(f"   {item} (directory)")
    
    for required_file in required_files:
        if not os.path.exists(os.path.join(model_path, required_file)):
            missing_files.append(required_file)
    
    if missing_files:
        print(f"❌ Missing required model files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Load the model
    print("🔄 Loading CodeBERT model...")
    try:
        codebert_trainer.load_model(model_path)
        print("✅ CodeBERT model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return False
    
    # Test the model
    test_code = "void test() { char buf[10]; strcpy(buf, input); }"
    print("🧪 Testing model prediction...")
    
    try:
        test_result = codebert_trainer.predict(test_code)
        print(f"✅ Model test successful!")
        print(f"   Test prediction: {test_result['prediction']} ({'Vulnerable' if test_result['prediction'] == 1 else 'Safe'})")
        print(f"   Test confidence: {test_result['confidence']:.4f}")
        print(f"   Test probabilities: {test_result['probabilities']}")
        
        # Test with a sample that should be vulnerable according to your model
        vulnerable_samples = [
            "int main() { char buf[10]; gets(buf); return 0; }",
            "void func(char *input) { strcpy(dest, input); }",
            "char* process(char* input) { char buffer[100]; sprintf(buffer, \"%s\", input); return buffer; }"
        ]
        
        print("\n🚨 Testing samples that should be vulnerable:")
        vulnerable_found = 0
        
        for i, sample in enumerate(vulnerable_samples):
            try:
                result = codebert_trainer.predict(sample)
                pred = result['prediction']
                conf = result['confidence']
                status = "Vulnerable" if pred == 1 else "Safe"
                print(f"   Sample {i+1}: {pred} ({status}) - Confidence: {conf:.4f}")
                if pred == 1:
                    vulnerable_found += 1
            except Exception as e:
                print(f"   Sample {i+1}: Error - {e}")
        
        print(f"\n📊 Found {vulnerable_found}/{len(vulnerable_samples)} samples predicted as vulnerable")
        
        if vulnerable_found == 0:
            print("⚠️ Warning: No samples predicted as vulnerable. Model might not be working as expected.")
        
        return True
        
    except Exception as e:
        print(f"❌ Model prediction test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_from_csv():
    """Test with a sample from the actual test CSV"""
    print(f"\n📊 Testing with sample from cwe399_test.csv:")
    print("=" * 50)
    
    try:
        import pandas as pd
        
        # Load a small sample from the test CSV
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cwe399_test.csv")
        if not os.path.exists(csv_path):
            print(f"❌ Test CSV not found: {csv_path}")
            return False
        
        # Read first 10 samples
        test_data = pd.read_csv(csv_path, nrows=10)
        print(f"✅ Loaded {len(test_data)} samples from CSV")
        
        # Initialize trainer and load model
        codebert_trainer = CodeBERTTrainer()
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(backend_dir), "codebert-model")
        codebert_trainer.load_model(model_path)
        
        vulnerable_count = 0
        
        print("\nTesting CSV samples:")
        for idx, row in test_data.iterrows():
            code = row['functionSource']
            true_label = row.get('label', 'Unknown')
            
            try:
                result = codebert_trainer.predict(code)
                pred = result['prediction']
                conf = result['confidence']
                
                if pred == 1:
                    vulnerable_count += 1
                    print(f"   Sample {idx}: VULNERABLE (confidence: {conf:.4f}, true: {true_label})")
                
            except Exception as e:
                print(f"   Sample {idx}: Error - {e}")
        
        print(f"\n📊 CSV Test Results: {vulnerable_count}/{len(test_data)} samples predicted as vulnerable")
        return True
        
    except Exception as e:
        print(f"❌ CSV test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Backend Model Loading Test")
    print("="*60)
    
    # Test backend model loading
    success = test_backend_model_loading()
    
    if success:
        print("\n✅ Backend model loading test PASSED!")
        
        # Test with CSV samples
        csv_success = test_sample_from_csv()
        
        if csv_success:
            print("\n🎉 All tests PASSED! Backend should work correctly.")
            print("\nNext steps:")
            print("1. Start the backend: cd backend && python main.py")
            print("2. Test via frontend or API calls")
        else:
            print("\n⚠️ CSV test failed, but basic model loading works")
    else:
        print("\n❌ Backend model loading test FAILED!")
        print("Please fix the issues above before starting the backend.") 