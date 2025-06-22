import pandas as pd
import os
from train_codebert_model import CodeBERTTrainer

def load_and_test_model_example():
    """
    Example of how to load a saved CodeBERT model and test it on data,
    showing which samples are predicted as vulnerable (label 1)
    """
    
    # 1. Initialize the trainer
    trainer = CodeBERTTrainer()
    
    # 2. Specify the path to your saved model
    # Replace this with the actual path to your saved model directory
    model_path = "codebert_outputs/model"  # Update this path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        print("Please update the model_path variable with the correct path to your saved model.")
        return
    
    try:
        # 3. Load the saved model
        print("🔄 Loading saved model...")
        trainer.load_model(model_path)
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # 4. Load your test data
    # Replace with your actual test data path or DataFrame
    # Expected format: DataFrame with 'functionSource' and optionally 'label' columns
    
    # Example 1: Load from CSV file
    test_data_path = "path/to/your/test_data.csv"  # Update this path
    
    # Example 2: Create sample test data (remove this when using real data)
    sample_test_data = pd.DataFrame({
        'functionSource': [
            'int main() { char buf[10]; gets(buf); return 0; }',  # Vulnerable - buffer overflow
            'int main() { int x = 5; return x; }',  # Not vulnerable
            'void func(char *input) { strcpy(dest, input); }',  # Vulnerable - strcpy
            'int add(int a, int b) { return a + b; }',  # Not vulnerable
            'char* process(char* input) { char buffer[100]; sprintf(buffer, "%s", input); return buffer; }'  # Vulnerable
        ],
        'label': [1, 0, 1, 0, 1]  # Optional: true labels for comparison
    })
    
    # Choose your data source:
    if os.path.exists(test_data_path):
        print(f"📁 Loading test data from: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
    else:
        print("📝 Using sample test data (update test_data_path for real data)")
        test_data = sample_test_data
    
    print(f"📊 Test data loaded: {len(test_data)} samples")
    
    # 5. Make predictions on the test data
    print("\n🔍 Making predictions...")
    results = trainer.predict_dataset(
        test_data, 
        dataset_name="test_analysis",
        export_predictions=True  # This will save predictions to a txt file
    )
    
    # 6. Show overall results
    print(f"\n📈 Prediction Results:")
    print(f"Total samples: {len(results['predictions'])}")
    
    vulnerable_count = sum(results['predictions'])
    not_vulnerable_count = len(results['predictions']) - vulnerable_count
    
    print(f"Predicted as Vulnerable (label 1): {vulnerable_count}")
    print(f"Predicted as Not Vulnerable (label 0): {not_vulnerable_count}")
    
    # Show accuracy if true labels are available
    if 'true_labels' in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
    
    # 7. Show samples predicted as vulnerable (label 1)
    print(f"\n🚨 SAMPLES PREDICTED AS VULNERABLE (Label 1):")
    print("=" * 80)
    
    vulnerable_samples = []
    for idx, (prediction, code) in enumerate(zip(results['predictions'], test_data['functionSource'])):
        if prediction == 1:  # Vulnerable prediction
            vulnerable_samples.append({
                'index': idx,
                'code': code,
                'prediction': prediction
            })
            
            print(f"\nSample #{idx}:")
            print(f"Prediction: {prediction} (Vulnerable)")
            
            # Show true label if available
            if 'label' in test_data.columns:
                true_label = test_data.iloc[idx]['label']
                correct = "✅" if prediction == true_label else "❌"
                print(f"True Label: {true_label} {correct}")
            
            print(f"Code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
    
    if not vulnerable_samples:
        print("No samples were predicted as vulnerable.")
    else:
        print(f"\nTotal vulnerable predictions: {len(vulnerable_samples)}")
    
    # 8. Show export path
    if 'export_path' in results:
        print(f"\n💾 Predictions exported to: {results['export_path']}")
    
    return results, vulnerable_samples

def test_single_code_sample():
    """
    Example of testing a single code sample
    """
    print("\n" + "="*60)
    print("TESTING SINGLE CODE SAMPLE")
    print("="*60)
    
    # Initialize trainer and load model
    trainer = CodeBERTTrainer()
    model_path = "codebert_outputs/model"  # Update this path
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return
    
    try:
        trainer.load_model(model_path)
        print("✅ Model loaded for single prediction test!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Test code sample (potentially vulnerable)
    test_code = """
    void vulnerable_function(char *user_input) {
        char buffer[100];
        strcpy(buffer, user_input);  // Vulnerable to buffer overflow
        printf("Input: %s", buffer);
    }
    """
    
    print(f"\n🧪 Testing code sample:")
    print("-" * 40)
    print(test_code)
    print("-" * 40)
    
    # Make prediction
    result = trainer.predict(test_code)
    
    print(f"\n📊 Prediction Results:")
    print(f"Prediction: {result['prediction']} ({result['label_names'][result['prediction']]})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities:")
    for i, (label, prob) in enumerate(zip(result['label_names'], result['probabilities'])):
        print(f"  {label}: {prob:.4f}")
    
    if result['prediction'] == 1:
        print("🚨 This code is predicted as VULNERABLE!")
    else:
        print("✅ This code is predicted as NOT VULNERABLE.")

if __name__ == "__main__":
    print("🚀 CodeBERT Model Testing Example")
    print("="*50)
    
    # Test on dataset
    try:
        results, vulnerable_samples = load_and_test_model_example()
    except Exception as e:
        print(f"❌ Error in dataset testing: {str(e)}")
    
    # Test single sample
    try:
        test_single_code_sample()
    except Exception as e:
        print(f"❌ Error in single sample testing: {str(e)}")
    
    print("\n✨ Testing complete!") 