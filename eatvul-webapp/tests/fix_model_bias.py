#!/usr/bin/env python3
"""
Fix CodeBERT model bias by retraining with class balancing techniques
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from train_codebert_model import CodeBERTTrainer, CodeDataset
import sys
import os

class BalancedCodeBERTTrainer(CodeBERTTrainer):
    """Enhanced CodeBERT trainer with bias correction techniques"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
    
    def calculate_class_weights(self, labels):
        """Calculate class weights to handle imbalance"""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Convert to tensor
        weight_tensor = torch.FloatTensor(class_weights)
        
        print(f"Class weights calculated:")
        for i, weight in enumerate(class_weights):
            print(f"  Class {i}: {weight:.4f}")
        
        return weight_tensor
    
    def train_model_balanced(self, data_loaders, freeze_bert=False):
        """Train model with balanced loss function"""
        
        # Calculate class weights from training data
        train_labels = []
        for batch in data_loaders['train_loader']:
            train_labels.extend(batch['label'].tolist())
        
        self.class_weights = self.calculate_class_weights(train_labels)
        
        # Count classes
        unique, counts = np.unique(train_labels, return_counts=True)
        total = len(train_labels)
        
        print(f"\nTraining data distribution:")
        for label, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
        
        imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 5:
            print("⚠️  Severe imbalance detected - applying strong balancing")
        
        # Initialize model
        from train_codebert_model import CodeBERTClassifier, device
        self.model = CodeBERTClassifier(freeze_bert=freeze_bert)
        self.model.to(device)
        
        # Define weighted loss function
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        
        # Optimizer with different learning rates for different parts
        optimizer_params = [
            {'params': self.model.codebert.parameters(), 'lr': self.learning_rate * 0.1},  # Lower LR for pretrained
            {'params': self.model.classifier.parameters(), 'lr': self.learning_rate}  # Higher LR for classifier
        ]
        
        from transformers import AdamW, get_linear_schedule_with_warmup
        optimizer = AdamW(optimizer_params)
        
        # Learning rate scheduler
        total_steps = len(data_loaders['train_loader']) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )
        
        # Training loop with bias correction
        print(f"\n=== Training Balanced CodeBERT Model ===")
        print(f"Using weighted CrossEntropyLoss with weights: {self.class_weights}")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            epoch_predictions = []
            epoch_labels = []
            
            from tqdm import tqdm
            progress_bar = tqdm(data_loaders['train_loader'], desc="Training")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate weighted loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Track training metrics
                train_loss += loss.item()
                
                # Get predictions for monitoring
                _, preds = torch.max(outputs, dim=1)
                epoch_predictions.extend(preds.cpu().tolist())
                epoch_labels.extend(labels.cpu().tolist())
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(data_loaders['train_loader'])
            
            # Calculate training class distribution for this epoch
            from sklearn.metrics import accuracy_score
            train_acc = accuracy_score(epoch_labels, epoch_predictions)
            
            unique_pred, pred_counts = np.unique(epoch_predictions, return_counts=True)
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Training predictions - Label 0: {pred_counts[0] if 0 in unique_pred else 0}, Label 1: {pred_counts[1] if 1 in unique_pred else 0}")
            
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_results = self.validate_epoch(data_loaders['val_loader'], criterion)
            
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_results['loss']:.4f}")
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
            print(f"Val predictions - Label 0: {val_results['pred_counts'][0]}, Label 1: {val_results['pred_counts'][1]}")
            
            # Save best model based on balanced accuracy
            val_accuracy = val_results['accuracy']
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best model with validation accuracy: {val_accuracy:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
        
        return self.model
    
    def validate_epoch(self, val_loader, criterion):
        """Validate one epoch and return metrics"""
        self.model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                from train_codebert_model import device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                val_predictions.extend(preds.cpu().tolist())
                val_true_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        avg_val_loss = val_loss / len(val_loader)
        
        # Count predictions
        unique_pred, pred_counts = np.unique(val_predictions, return_counts=True)
        pred_count_dict = {0: 0, 1: 0}
        for label, count in zip(unique_pred, pred_counts):
            pred_count_dict[label] = count
        
        # Store in history
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        
        return {
            'loss': avg_val_loss,
            'accuracy': val_accuracy,
            'predictions': val_predictions,
            'true_labels': val_true_labels,
            'pred_counts': pred_count_dict
        }

def create_balanced_sample_data():
    """Create a small balanced dataset for testing the fix"""
    
    # Vulnerable code samples
    vulnerable_codes = [
        "char buffer[10]; gets(buffer);",
        "system(user_input);",
        "strcpy(dest, src);",
        "sprintf(query, \"SELECT * FROM users WHERE name='%s'\", user_input);",
        "char *ptr = malloc(100); free(ptr); free(ptr);",
        "scanf(\"%s\", buffer);",
        "strcat(dest, src);",
        "void vulnerable() { char buf[10]; strcpy(buf, argv[1]); }",
        "int fd = open(filename, O_RDWR); write(fd, user_data, strlen(user_data));",
        "execl(\"/bin/sh\", \"-c\", user_command, NULL);",
    ] * 10  # Repeat to get more samples
    
    # Safe code samples  
    safe_codes = [
        "int x = 5; int y = x + 10; return y;",
        "if (input != NULL && strlen(input) < 100) { strncpy(buffer, input, 99); }",
        "const char* message = \"Hello World\"; printf(\"%s\", message);",
        "int array[10]; for(int i = 0; i < 10; i++) { array[i] = i; }",
        "FILE* file = fopen(\"data.txt\", \"r\"); if(file) { fclose(file); }",
        "char *safe_ptr = malloc(100); if(safe_ptr) { strcpy(safe_ptr, \"safe\"); free(safe_ptr); }",
        "int len = strlen(input); if(len < MAX_SIZE) { memcpy(buffer, input, len); }",
        "void safe_function() { const int SIZE = 100; char buf[SIZE]; snprintf(buf, SIZE, \"safe\"); }",
        "size_t len = strnlen(src, MAX_LEN); if(len < MAX_LEN) { strncpy(dest, src, len); }",
        "if(validate_input(user_data)) { process_safe_data(user_data); }",
    ] * 10  # Repeat to get more samples
    
    # Create balanced dataset
    all_codes = vulnerable_codes + safe_codes
    all_labels = [1] * len(vulnerable_codes) + [0] * len(safe_codes)
    
    # Create DataFrame
    df = pd.DataFrame({
        'functionSource': all_codes,
        'label': all_labels
    })
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Created balanced dataset with {len(df)} samples")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    return df

def retrain_with_bias_correction():
    """Retrain the model with bias correction techniques"""
    
    print("🔧 RETRAINING MODEL WITH BIAS CORRECTION")
    print("="*60)
    
    # Create or load balanced training data
    print("Creating balanced sample dataset...")
    sample_data = create_balanced_sample_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        sample_data, 
        test_size=0.2, 
        stratify=sample_data['label'],
        random_state=42
    )
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    # Initialize balanced trainer
    trainer = BalancedCodeBERTTrainer(
        batch_size=4,  # Smaller batch size for stability
        epochs=3,      # Fewer epochs for quick testing
        learning_rate=5e-5  # Lower learning rate
    )
    
    # Prepare data loaders
    data_loaders = trainer.prepare_data(train_data, val_data)
    
    # Train with balance correction
    print("\nStarting balanced training...")
    model = trainer.train_model_balanced(data_loaders)
    
    # Test the retrained model
    print("\n🧪 TESTING RETRAINED MODEL")
    print("-" * 40)
    
    # Test on the same samples that failed before
    test_samples = [
        ("char buffer[10]; gets(buffer);", 1),
        ("system(user_input);", 1),  
        ("strcpy(dest, src);", 1),
        ("int x = 5; int y = x + 10; return y;", 0),
        ("const char* message = \"Hello\"; printf(\"%s\", message);", 0)
    ]
    
    correct_predictions = 0
    for code, expected in test_samples:
        result = trainer.predict(code)
        prediction = result['prediction']
        confidence = result['confidence']
        
        is_correct = prediction == expected
        if is_correct:
            correct_predictions += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {code[:50]}...")
        print(f"   Expected: {expected}, Got: {prediction}, Confidence: {confidence:.4f}")
    
    accuracy = correct_predictions / len(test_samples)
    print(f"\nTest accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_samples)})")
    
    if accuracy > 0.6:
        print("🎉 Model bias significantly improved!")
        
        # Save the retrained model
        model_dir = trainer.save_model()
        print(f"Improved model saved to: {model_dir}")
        
        return trainer
    else:
        print("⚠️  Model still has bias issues. Consider:")
        print("   1. More balanced training data")
        print("   2. Different class weights")
        print("   3. Data augmentation techniques")
        
        return None

if __name__ == "__main__":
    retrain_with_bias_correction() 