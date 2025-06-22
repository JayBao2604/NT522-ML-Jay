#!/usr/bin/env python3
"""
Test script to verify model loading and predictions using local model and test data
This replicates the user's working script to verify functionality
"""
import pandas as pd
import os
from train_codebert_model import CodeBERTTrainer

def test_local_model_prediction():
    """
    Test the local model with the cwe399_test.csv data to verify predictions
    """
    print("🔍 Testing Local Model Prediction")
    print("=" * 50)
    
    # Load test data from the local file
    test_data_path = "cwe399_test.csv"
    
    if not os.path.exists(test_data_path):
        print(f"❌ Test data file not found: {test_data_path}")
        return False
    
    print(f"📁 Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    print(f"✅ Test data loaded: {len(test_data)} samples")
    print(f"📋 Columns: {list(test_data.columns)}")
    
    # Show a sample of the data
    print(f"\n📊 Data sample:")
    if 'functionSource' in test_data.columns:
        print(f"First functionSource sample (first 200 chars):")
        print(f"{test_data['functionSource'].iloc[0][:200]}...")
    if 'label' in test_data.columns:
        print(f"Label distribution: {test_data['label'].value_counts().to_dict()}")
    
    # Initialize trainer
    print(f"\n🔄 Initializing CodeBERT trainer...")
    trainer = CodeBERTTrainer()
    print("✅ CodeBERTTrainer initialized")
    
    # Use the local model path (same as backend)
    model_path = "codebert-model"
    
    print(f"\n🔄 Loading saved model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    try:
        trainer.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Make predictions on the test data
    print("\n🔍 Making predictions...")
    try:
        results = trainer.predict_dataset(
            test_data, 
            dataset_name="cwe399_local_test",
            export_predictions=True  # This will save predictions to a txt file
        )
        print("✅ Predictions completed successfully!")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Show overall results
    print(f"\n📈 Prediction Results:")
    print(f"Total samples: {len(results['predictions'])}")
    
    vulnerable_count = sum(results['predictions'])
    not_vulnerable_count = len(results['predictions']) - vulnerable_count
    
    print(f"Predicted as Vulnerable (label 1): {vulnerable_count}")
    print(f"Predicted as Not Vulnerable (label 0): {not_vulnerable_count}")
    print(f"Vulnerability rate: {vulnerable_count/len(results['predictions'])*100:.1f}%")
    
    # Show accuracy if true labels are available
    if 'true_labels' in results:
        print(f"\n📊 Performance Metrics:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
    
    # Show first few samples predicted as vulnerable (label 1)
    print(f"\n🚨 SAMPLES PREDICTED AS VULNERABLE (Label 1) - First 5:")
    print("=" * 80)
    
    vulnerable_samples = []
    vulnerable_count_shown = 0
    max_show = 5  # Limit output for readability
    
    for idx, (prediction, code) in enumerate(zip(results['predictions'], test_data['functionSource'])):
        if prediction == 1:  # Vulnerable prediction
            vulnerable_samples.append({
                'index': idx,
                'code': code,
                'prediction': prediction
            })
            
            if vulnerable_count_shown < max_show:
                print(f"\nSample #{idx}:")
                print(f"Prediction: {prediction} (Vulnerable)")
                
                # Show true label if available
                if 'label' in test_data.columns:
                    true_label = test_data.iloc[idx]['label']
                    correct = "✅" if prediction == true_label else "❌"
                    print(f"True Label: {true_label} {correct}")
                
                print(f"Code (first 300 chars):")
                print("-" * 40)
                print(code[:300] + ("..." if len(code) > 300 else ""))
                print("-" * 40)
                vulnerable_count_shown += 1
    
    if not vulnerable_samples:
        print("❌ No samples were predicted as vulnerable.")
        print("⚠️  This suggests the model might not be working correctly.")
    else:
        print(f"\n📊 Summary: {len(vulnerable_samples)} total vulnerable predictions")
        if len(vulnerable_samples) > max_show:
            print(f"   (Showing first {max_show} samples above)")
    
    # Show export path
    if 'export_path' in results:
        print(f"\n💾 Predictions exported to: {results['export_path']}")
    
    # Test a few individual samples to verify prediction function works
    print(f"\n🧪 Testing individual sample predictions:")
    test_samples = [
        "void test() { char buf[10]; strcpy(buf, input); }",  # Should be vulnerable
        "void safe() { printf(\"Hello World\"); }",  # Should be safe
        "int main() { char buffer[10]; gets(buffer); return 0; }"  # Should be vulnerable
    ]
    
    for i, sample in enumerate(test_samples):
        try:
            result = trainer.predict(sample)
            pred = result['prediction']
            conf = result['confidence']
            print(f"  Sample {i+1}: {pred} ({'Vulnerable' if pred == 1 else 'Safe'}) - Confidence: {conf:.4f}")
        except Exception as e:
            print(f"  Sample {i+1}: Error - {e}")
    
    return True

def compare_with_backend_path():
    """
    Test if the backend can find and load the model correctly
    """
    print(f"\n🔧 Testing Backend Model Path Resolution:")
    print("=" * 50)
    
    # Simulate backend path resolution
    backend_dir = os.path.join(os.getcwd(), "backend")
    model_path = os.path.join(os.path.dirname(backend_dir), "codebert-model")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Backend directory: {backend_dir}")
    print(f"Backend resolved model path: {model_path}")
    print(f"Model path exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print("✅ Backend should be able to find the model")
        
        # List model files
        print("📂 Model files:")
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"   {item} ({size_mb:.1f} MB)")
        
        return True
    else:
        print("❌ Backend will not be able to find the model")
        return False

if __name__ == "__main__":
    print("🚀 Local Model and Data Testing")
    print("="*60)
    
    # Test local model prediction
    success = test_local_model_prediction()
    
    # Test backend path resolution
    backend_path_ok = compare_with_backend_path()
    
    print(f"\n🎯 Test Results:")
    print(f"   Local model prediction: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Backend path resolution: {'✅ PASS' if backend_path_ok else '❌ FAIL'}")
    
    if success and backend_path_ok:
        print(f"\n✅ All tests passed! The model should work correctly.")
        print(f"   If frontend still shows wrong results, the issue is likely in:")
        print(f"   1. Backend model loading during startup")
        print(f"   2. Backend falling back to keyword-based analysis")
        print(f"   3. API communication between frontend and backend")
    elif success and not backend_path_ok:
        print(f"\n⚠️  Model works locally but backend path resolution failed.")
        print(f"   Check the backend model path configuration.")
    else:
        print(f"\n❌ Model loading/prediction failed.")
        print(f"   Please check model files and dependencies.") 