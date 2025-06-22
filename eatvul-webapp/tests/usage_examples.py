"""
Usage Examples for CodeBERT Training Code
==========================================

This file demonstrates how to use the CodeBERTTrainer class for various tasks:
1. Training a new model
2. Loading and evaluating a saved model  
3. Making predictions on new data
4. Step-by-step training process
"""

import pandas as pd
from train_codebert_model import CodeBERTTrainer, free_gpu_memory

# =============================================================================
# Example 1: Complete Training Pipeline (Recommended for beginners)
# =============================================================================

def example_1_complete_training():
    """
    Train a CodeBERT model using the simplified run_all method.
    This handles data loading, training, evaluation, and saving automatically.
    """
    print("=== Example 1: Complete Training Pipeline ===")
    
    # Method 1a: Load data from file
    trainer = CodeBERTTrainer(
        batch_size=8,      # Adjust based on your GPU memory
        epochs=4,          # Number of training epochs
        learning_rate=2e-5 # Learning rate for AdamW optimizer
    )
    
    # If you have separate train/test files
    train_data = pd.read_csv('train_data.csv')  # Must have 'functionSource' and 'label' columns
    test_data = pd.read_csv('test_data.csv')    # Must have 'functionSource' and 'label' columns
    
    # Train and evaluate the model
    results = trainer.run_all(
        train_data=train_data,
        test_data=test_data,
        freeze_bert=False,      # Set to True to freeze CodeBERT layers
        dataset_name="my_dataset"  # Used for naming output files
    )
    
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1']:.4f}")
    
    # Free GPU memory after training
    free_gpu_memory()

# =============================================================================
# Example 2: Training with DataFrame Input
# =============================================================================

def example_2_dataframe_training():
    """
    Train using pandas DataFrames directly (useful for preprocessed data)
    """
    print("=== Example 2: Training with DataFrames ===")
    
    # Create or load your data as DataFrames
    # Your data must have these columns: 'functionSource' (code) and 'label' (0 or 1)
    train_df = pd.DataFrame({
        'functionSource': [
            'int vulnerable_func(char *input) { char buffer[10]; strcpy(buffer, input); return 0; }',
            'int safe_func(char *input) { char buffer[100]; strncpy(buffer, input, 99); buffer[99] = 0; return 0; }'
        ],
        'label': [1, 0]  # 1 = vulnerable, 0 = not vulnerable
    })
    
    test_df = pd.DataFrame({
        'functionSource': [
            'void test_func(char *data) { char buf[50]; sprintf(buf, "%s", data); }',
        ],
        'label': [1]
    })
    
    trainer = CodeBERTTrainer(batch_size=4, epochs=2)
    
    results = trainer.run_all(
        train_data=train_df,
        test_data=test_df,
        dataset_name="example_dataset"
    )
    
    return results

# =============================================================================
# Example 3: Step-by-Step Training (Advanced)
# =============================================================================

def example_3_manual_training():
    """
    Manual step-by-step training for more control over the process
    """
    print("=== Example 3: Manual Step-by-Step Training ===")
    
    # Initialize trainer
    trainer = CodeBERTTrainer(batch_size=8, epochs=3, learning_rate=1e-5)
    
    # Load your data
    train_data = pd.read_csv('your_train_data.csv')
    test_data = pd.read_csv('your_test_data.csv')
    
    # Prepare data loaders
    data_loaders = trainer.prepare_data(train_data, test_data)
    
    # Train the model
    model = trainer.train_model(data_loaders, freeze_bert=False)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate the model
    results = trainer.evaluate_model(data_loaders['test_loader'], dataset_name="manual_test")
    
    # Save the model
    model_dir = trainer.save_model()
    
    print(f"Model saved to: {model_dir}")
    return results

# =============================================================================
# Example 4: Loading and Using a Saved Model
# =============================================================================

def example_4_load_saved_model():
    """
    Load a previously trained model and use it for evaluation or prediction
    """
    print("=== Example 4: Loading Saved Model ===")
    
    trainer = CodeBERTTrainer()
    
    # Load a saved model (replace with your actual model directory)
    model_dir = "codebert_outputs/model"
    trainer.load_model(model_dir)
    
    # Method 4a: Evaluate on new test data
    test_data = pd.read_csv('new_test_data.csv')
    results = trainer.evaluate_saved_model(
        model_dir=model_dir,
        test_data=test_data,
        dataset_name="new_test",
        export_predictions=True
    )
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1: {results['f1']:.4f}")
    
    return results

# =============================================================================
# Example 5: Making Predictions on New Data
# =============================================================================

def example_5_predictions():
    """
    Use a trained model to make predictions on new data
    """
    print("=== Example 5: Making Predictions ===")
    
    trainer = CodeBERTTrainer()
    
    # Load your trained model
    model_dir = "codebert_outputs/model"
    trainer.load_model(model_dir)
    
    # Method 5a: Predict on a dataset
    new_data = pd.DataFrame({
        'functionSource': [
            'void func1(char *input) { char buf[10]; strcpy(buf, input); }',
            'void func2(char *input) { char buf[100]; strncpy(buf, input, 99); buf[99] = 0; }'
        ],
        'label': [1, 0]  # Optional: include if you want to calculate accuracy
    })
    
    prediction_results = trainer.predict_dataset(
        test_data=new_data,
        dataset_name="new_predictions",
        export_predictions=True
    )
    
    print("Predictions:", prediction_results['predictions'])
    
    # Method 5b: Predict on a single code sample
    code_sample = """
    int process_input(char *user_input) {
        char buffer[50];
        strcpy(buffer, user_input);  // Potential buffer overflow
        return strlen(buffer);
    }
    """
    
    single_prediction = trainer.predict(code_sample)
    print(f"Single prediction: {single_prediction['prediction']}")
    print(f"Confidence: {single_prediction['confidence']:.4f}")
    print(f"Label: {single_prediction['label_names'][single_prediction['prediction']]}")
    
    return prediction_results

# =============================================================================
# Example 6: Data Format Requirements
# =============================================================================

def example_6_data_format():
    """
    Shows the required data format for training
    """
    print("=== Example 6: Required Data Format ===")
    
    # Your CSV/Excel/JSON files must have these columns:
    sample_data = pd.DataFrame({
        'functionSource': [
            # C/C++ code snippets as strings
            'int func1(char *input) { char buf[10]; strcpy(buf, input); return 0; }',
            'int func2(char *input) { if(strlen(input) < 50) strcpy(buf, input); return 0; }',
            'int func3(char *input) { char buf[100]; strncpy(buf, input, 99); return 0; }'
        ],
        'label': [
            1,  # Vulnerable (has strcpy without bounds checking)
            1,  # Vulnerable (buf not declared in scope shown)  
            0   # Not vulnerable (uses strncpy with bounds checking)
        ]
    })
    
    print("Required columns:")
    print("- 'functionSource': String containing the code to analyze")
    print("- 'label': Integer (0 = not vulnerable, 1 = vulnerable)")
    print("\nExample data structure:")
    print(sample_data)
    
    # Save example data
    sample_data.to_csv('example_training_data.csv', index=False)
    print("\nSaved example data to 'example_training_data.csv'")

# =============================================================================
# Example 7: Configuration Options
# =============================================================================

def example_7_configuration():
    """
    Shows different configuration options
    """
    print("=== Example 7: Configuration Options ===")
    
    # Different configurations for different use cases
    
    # Configuration 1: Quick testing (small batch, few epochs)
    quick_trainer = CodeBERTTrainer(
        batch_size=4,       # Small batch for limited GPU memory
        epochs=2,           # Few epochs for quick testing
        learning_rate=2e-5  # Standard learning rate
    )
    
    # Configuration 2: Full training (larger batch, more epochs)
    full_trainer = CodeBERTTrainer(
        batch_size=16,      # Larger batch if you have sufficient GPU memory
        epochs=10,          # More epochs for better training
        learning_rate=1e-5  # Lower learning rate for fine-tuning
    )
    
    # Configuration 3: Fine-tuning existing model (freeze base layers)
    finetune_trainer = CodeBERTTrainer(
        batch_size=8,
        epochs=5,
        learning_rate=3e-5
    )
    
    print("Configuration options:")
    print("- batch_size: 4-16 (adjust based on GPU memory)")
    print("- epochs: 2-10 (more epochs = better training but longer time)")
    print("- learning_rate: 1e-5 to 3e-5 (lower = more stable, higher = faster)")
    print("- freeze_bert: True/False (freeze CodeBERT layers for faster training)")

# =============================================================================
# Main execution examples
# =============================================================================

if __name__ == "__main__":
    # Run individual examples (uncomment the ones you want to try)
    
    # Show data format requirements first
    example_6_data_format()
    
    # Show configuration options
    example_7_configuration()
    
    # For actual training, uncomment one of these:
    # example_1_complete_training()
    # example_2_dataframe_training()
    # example_3_manual_training()
    
    # For using saved models, uncomment these:
    # example_4_load_saved_model()
    # example_5_predictions()
    
    print("\n=== Usage Examples Complete ===")
    print("Uncomment the examples you want to run in the if __name__ == '__main__' section") 