#!/usr/bin/env python3
"""
Diagnostic script to test CodeBERT model predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_codebert_model import CodeBERTTrainer

def test_model_predictions():
    """Test model predictions on clearly vulnerable and safe code"""
    
    # Test samples - clearly vulnerable
    vulnerable_samples = [
        "char buffer[10]; gets(buffer);",  # Buffer overflow
        "system(user_input);",  # Command injection
        "strcpy(dest, src);",  # No bounds checking
        "sprintf(query, \"SELECT * FROM users WHERE name='%s'\", user_input);",  # SQL injection
        "char *ptr = malloc(100); free(ptr); free(ptr);",  # Double free
    ]
    
    # Test samples - clearly safe
    safe_samples = [
        "int x = 5; int y = x + 10; return y;",  # Simple arithmetic
        "if (input != NULL && strlen(input) < 100) { strncpy(buffer, input, 99); }",  # Safe copy
        "const char* message = \"Hello World\"; printf(\"%s\", message);",  # Safe print
        "int array[10]; for(int i = 0; i < 10; i++) { array[i] = i; }",  # Safe loop
        "FILE* file = fopen(\"data.txt\", \"r\"); if(file) { fclose(file); }",  # Safe file handling
    ]
    
    print("="*60)
    print("CODEBERT MODEL PREDICTION DIAGNOSTIC")
    print("="*60)
    
    try:
        # Load model
        trainer = CodeBERTTrainer()
        model_path = os.path.join("..", "codebert-model")
        
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            return
            
        print(f"Loading model from: {model_path}")
        trainer.load_model(model_path)
        print("✅ Model loaded successfully")
        
        print("\n🔴 TESTING CLEARLY VULNERABLE CODE:")
        print("-" * 40)
        
        vulnerable_results = []
        for i, code in enumerate(vulnerable_samples, 1):
            try:
                result = trainer.predict(code)
                vulnerable_results.append(result)
                
                print(f"\n[{i}] Code: {code}")
                print(f"    Prediction: {result['prediction']} ({'Vulnerable' if result['prediction'] == 1 else 'Safe'})")
                print(f"    Confidence: {result['confidence']:.4f}")
                print(f"    Probabilities: [Safe: {result['probabilities'][0]:.4f}, Vuln: {result['probabilities'][1]:.4f}]")
                
            except Exception as e:
                print(f"[{i}] ERROR: {str(e)}")
        
        print("\n🟢 TESTING CLEARLY SAFE CODE:")
        print("-" * 40)
        
        safe_results = []
        for i, code in enumerate(safe_samples, 1):
            try:
                result = trainer.predict(code)
                safe_results.append(result)
                
                print(f"\n[{i}] Code: {code}")
                print(f"    Prediction: {result['prediction']} ({'Vulnerable' if result['prediction'] == 1 else 'Safe'})")
                print(f"    Confidence: {result['confidence']:.4f}")
                print(f"    Probabilities: [Safe: {result['probabilities'][0]:.4f}, Vuln: {result['probabilities'][1]:.4f}]")
                
            except Exception as e:
                print(f"[{i}] ERROR: {str(e)}")
        
        # Analysis
        print("\n📊 ANALYSIS:")
        print("=" * 40)
        
        vuln_predictions = [r['prediction'] for r in vulnerable_results]
        safe_predictions = [r['prediction'] for r in safe_results]
        
        vuln_correct = sum([1 for p in vuln_predictions if p == 1])
        safe_correct = sum([1 for p in safe_predictions if p == 0])
        
        print(f"Vulnerable samples correctly identified: {vuln_correct}/{len(vulnerable_results)} ({vuln_correct/len(vulnerable_results)*100:.1f}%)")
        print(f"Safe samples correctly identified: {safe_correct}/{len(safe_results)} ({safe_correct/len(safe_results)*100:.1f}%)")
        
        # Check for bias
        all_predictions = vuln_predictions + safe_predictions
        label_0_count = sum([1 for p in all_predictions if p == 0])
        label_1_count = sum([1 for p in all_predictions if p == 1])
        
        print(f"\nPrediction distribution:")
        print(f"  Label 0 (Safe): {label_0_count}/{len(all_predictions)} ({label_0_count/len(all_predictions)*100:.1f}%)")
        print(f"  Label 1 (Vulnerable): {label_1_count}/{len(all_predictions)} ({label_1_count/len(all_predictions)*100:.1f}%)")
        
        if label_0_count == len(all_predictions):
            print("\n⚠️  MODEL BIAS DETECTED:")
            print("   The model is predicting ONLY label 0 (Safe)")
            print("   This indicates a severe bias issue!")
            
        # Check confidence levels
        all_confidences = [r['confidence'] for r in vulnerable_results + safe_results]
        avg_confidence = sum(all_confidences) / len(all_confidences)
        print(f"\nAverage confidence: {avg_confidence:.4f}")
        
        if avg_confidence > 0.95:
            print("⚠️  HIGH CONFIDENCE BIAS: Model is overly confident")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_predictions() 