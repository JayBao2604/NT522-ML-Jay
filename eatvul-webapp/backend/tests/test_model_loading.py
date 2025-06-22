#!/usr/bin/env python3
"""
Test script to diagnose model loading issues in the backend
"""
import os
import sys
import traceback

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_codebert_model import CodeBERTTrainer

def test_model_loading():
    """Test if the model loads correctly in the backend environment"""
    print("=== Backend Model Loading Test ===\n")
    
    # Test the exact path used by the backend
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(backend_dir), "codebert-model")
    
    print(f"Backend directory: {backend_dir}")
    print(f"Model path: {model_path}")
    print(f"Model path exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"Contents of model directory:")
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
            print(f"  {item} ({size} bytes)")
    
    print("\n" + "="*50)
    print("Testing model loading...")
    
    try:
        # Initialize trainer
        trainer = CodeBERTTrainer()
        print("✅ CodeBERTTrainer initialized successfully")
        
        # Load model
        print(f"Loading model from: {model_path}")
        trainer.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Test prediction with a simple vulnerable code sample
        test_code = '''
void vulnerable_function(char* input) {
    char buffer[10];
    strcpy(buffer, input);  // Buffer overflow vulnerability
    printf("%s", buffer);
}
'''
        
        print("\nTesting prediction...")
        result = trainer.predict(test_code)
        
        print(f"✅ Prediction successful!")
        print(f"   Prediction: {result['prediction']} ({'Vulnerable' if result['prediction'] == 1 else 'Safe'})")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Probabilities: {result['probabilities']}")
        
        # Test another sample that should be safe
        safe_code = '''
void safe_function(char* input) {
    char buffer[100];
    if (strlen(input) < sizeof(buffer)) {
        strncpy(buffer, input, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\\0';
    }
    printf("%s", buffer);
}
'''
        
        print("\nTesting with safe code...")
        safe_result = trainer.predict(safe_code)
        
        print(f"✅ Safe code prediction:")
        print(f"   Prediction: {safe_result['prediction']} ({'Vulnerable' if safe_result['prediction'] == 1 else 'Safe'})")
        print(f"   Confidence: {safe_result['confidence']:.4f}")
        print(f"   Probabilities: {safe_result['probabilities']}")
        
        return True, trainer
        
    except Exception as e:
        print(f"❌ Error during model loading/testing:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        traceback.print_exc()
        return False, None

def test_backend_imports():
    """Test if all backend imports work correctly"""
    print("\n=== Testing Backend Imports ===\n")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        from transformers import RobertaTokenizerFast, RobertaModel
        print("✅ RoBERTa components imported successfully")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Backend Model Diagnostic Test")
    print("="*50)
    
    # Test imports
    if not test_backend_imports():
        print("❌ Import test failed. Please fix import issues first.")
        return
    
    # Test model loading
    success, trainer = test_model_loading()
    
    if success:
        print("\n✅ All tests passed! Model should work in backend.")
        print("\nTo verify in backend:")
        print("1. Start the backend: cd backend && python main.py")
        print("2. Check diagnostic endpoint: http://localhost:3001/diagnostic")
        print("3. Test vulnerability analysis through the API")
    else:
        print("\n❌ Model loading failed. Please check the error above.")
        print("\nPossible solutions:")
        print("1. Ensure the model files are in the correct location")
        print("2. Check if the model was saved correctly")
        print("3. Verify all dependencies are installed")
        print("4. Try loading the model manually to check for corruption")

if __name__ == "__main__":
    main() 