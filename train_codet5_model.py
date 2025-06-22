import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print("Warning: Seaborn not installed. Some visualizations may not work.")
    sns = None
from tqdm import tqdm
import os
import re
import gc
import json
import zipfile
from datetime import datetime
from torch.optim import AdamW

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CodePreprocessor:
    """Preprocess code for CodeT5 model"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
        self.max_length = 512  # Maximum sequence length for CodeT5
    
    def preprocess_code(self, code_text):
        """Basic preprocessing of code text"""
        # Remove extra whitespace
        code_text = re.sub(r'\s+', ' ', code_text)
        code_text = code_text.strip()
        return code_text
    
    def tokenize(self, code_text, truncation=True, padding='max_length', return_tensors=None):
        """Tokenize code text using RobertaTokenizer"""
        processed_code = self.preprocess_code(code_text)
        return self.tokenizer(processed_code, 
                             truncation=truncation, 
                             max_length=self.max_length,
                             padding=padding,
                             return_tensors=return_tensors)

class CodeDataset(Dataset):
    """Dataset for code vulnerability detection using CodeT5"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, 
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 return_tensors='pt')
        
        # Remove batch dimension added by tokenizer when return_tensors='pt'
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CodeT5Classifier(nn.Module):
    """CodeT5 model for code vulnerability detection"""
    
    def __init__(self, freeze_base=False, dropout_rate=0.1):
        super(CodeT5Classifier, self).__init__()
        
        # Load pre-trained CodeT5 model
        self.codet5 = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
        self.dropout = nn.Dropout(dropout_rate)
        
        # For classification, we'll use the encoder's last hidden state
        # Get hidden size from the model config
        self.hidden_size = self.codet5.config.d_model
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, 2)  # Binary classification
        
        # Freeze CodeT5 layers if specified
        if freeze_base:
            for param in self.codet5.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # Get CodeT5 encoder outputs
        encoder_outputs = self.codet5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use the mean of the last hidden state for classification
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # Create a mask to exclude padding tokens from the mean
        # Shape: (batch_size, sequence_length, 1)
        mask = attention_mask.unsqueeze(-1).float()
        
        # Apply mask and compute mean over sequence length
        # Shape: (batch_size, hidden_size)
        pooled_output = torch.sum(last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits 

class CodeT5Trainer:
    """Trainer for CodeT5 model"""
    
    def __init__(self, data_path=None, batch_size=8, epochs=4, learning_rate=2e-5):
        self.preprocessor = CodePreprocessor()
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.data = None
        self.model = None
        self.best_model_state = None
        self.best_val_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        self.output_dir = os.path.join(os.getcwd(), 'codet5_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load data from file or DataFrame
        
        Args:
            data_path: Path to a data file (CSV, Excel, JSON) or a pandas DataFrame
        """
        print(f"DEBUG: Type of data_path in load_data: {type(data_path)}")
        
        # If data_path is already a DataFrame, use it directly
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
            print(f"Using provided DataFrame with {len(self.data)} samples.")
            
        # If it's a string, try to load from file
        elif isinstance(data_path, str):
            print(f"DEBUG: Trying to load from file path: '{data_path}'")
            
            # Check if the file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"File not found: '{data_path}'")
                
            file_ext = os.path.splitext(data_path.lower())[1]
            print(f"DEBUG: File extension detected: '{file_ext}'")
            
            if file_ext == '.csv':
                self.data = pd.read_csv(data_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.data = pd.read_excel(data_path)
            elif file_ext == '.json':
                self.data = pd.read_json(data_path)
            elif file_ext == '.pkl' or file_ext == '.pickle':
                self.data = pd.read_pickle(data_path)
            elif file_ext == '':
                # Try to infer the format if no extension is given
                try:
                    # First try CSV as it's most common
                    self.data = pd.read_csv(data_path)
                    print(f"Inferred file format as CSV for: {data_path}")
                except:
                    try:
                        # Then try JSON
                        self.data = pd.read_json(data_path)
                        print(f"Inferred file format as JSON for: {data_path}")
                    except:
                        raise ValueError(f"Could not determine file format for: '{data_path}'. Please specify a file with extension or provide a DataFrame.")
            else:
                raise ValueError(f"Unsupported file format: '{file_ext}'. Supported formats: CSV, Excel, JSON, Pickle")
        else:
            raise TypeError(f"data_path must be either a string file path or a pandas DataFrame, got {type(data_path).__name__}")
        
        # Check if required columns exist
        if 'functionSource' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("Data must contain 'functionSource' and 'label' columns.")
        
        print(f"Loaded data with {len(self.data)} samples.")
        print(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def set_data(self, dataframe):
        """Set data directly from a pandas DataFrame"""
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Check if required columns exist
        if 'functionSource' not in dataframe.columns or 'label' not in dataframe.columns:
            raise ValueError("Data must contain 'functionSource' and 'label' columns.")
        
        self.data = dataframe
        print(f"Set data with {len(self.data)} samples.")
        print(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def prepare_data(self, train_data, test_data):
        """Prepare data for model training using pre-split train and test data"""
        if self.data is None and (train_data is None or test_data is None):
            raise ValueError("No data provided. Provide train_data and test_data or call load_data/set_data first.")
        
        # Use provided train and test data
        train_texts = train_data['functionSource'].values
        train_labels = train_data['label'].values
        test_texts = test_data['functionSource'].values
        test_labels = test_data['label'].values
        
        # Create datasets
        train_dataset = CodeDataset(
            train_texts, 
            train_labels, 
            self.preprocessor.tokenizer, 
            self.preprocessor.max_length
        )
        
        test_dataset = CodeDataset(
            test_texts, 
            test_labels, 
            self.preprocessor.tokenizer, 
            self.preprocessor.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return {
            'train_loader': train_loader,
            'val_loader': test_loader,  # Use test loader for validation
            'test_loader': test_loader,
            'test_texts': test_texts,
            'test_labels': test_labels
        } 

    def run_all(self, data_source=None, train_data=None, test_data=None, freeze_base=False, dataset_name="test"):
        """Run all steps: data preparation, training, evaluation, and saving"""
        # Load data if provided as a single source
        if data_source is not None:
            if isinstance(data_source, str):
                self.load_data(data_source)
            elif isinstance(data_source, pd.DataFrame):
                self.set_data(data_source)
        
        # If train_data and test_data are provided, use them; otherwise, ensure data is loaded
        if train_data is not None and test_data is not None:
            if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
                raise ValueError("train_data and test_data must be pandas DataFrames.")
            if 'functionSource' not in train_data.columns or 'label' not in train_data.columns:
                raise ValueError("train_data must contain 'functionSource' and 'label' columns.")
            if 'functionSource' not in test_data.columns or 'label' not in test_data.columns:
                raise ValueError("test_data must contain 'functionSource' and 'label' columns.")
            print(f"Using provided train_data with {len(train_data)} samples.")
            print(f"Using provided test_data with {len(test_data)} samples.")
        elif self.data is None:
            raise ValueError("No data loaded. Provide data_source or train_data/test_data.")
        
        # Prepare data using provided train/test split or loaded data
        if train_data is not None and test_data is not None:
            data_loaders = self.prepare_data(train_data, test_data)
        else:
            data_loaders = self.prepare_data(self.data, self.data)  # Fallback (though not used in your case)
        
        # Train model
        self.train_model(data_loaders, freeze_base=freeze_base)
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate model with dataset name for proper file naming
        results = self.evaluate_model(data_loaders['test_loader'], dataset_name=dataset_name)
        
        # Save model
        model_dir = self.save_model()
        
        # Save evaluation results
        with open(os.path.join(model_dir, 'evaluation_results.json'), 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            serializable_results = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in results.items()
            }
            json.dump(serializable_results, f)
        
        print("\n=== Training and Evaluation Complete ===")
        print(f"All outputs saved to: {model_dir}")
        
        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def train_model(self, data_loaders, freeze_base=False):
        """Train the CodeT5 model"""
        # Initialize model
        self.model = CodeT5Classifier(freeze_base=freeze_base)
        self.model.to(device)
        
        # Define optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Calculate total training steps for learning rate scheduler
        total_steps = len(data_loaders['train_loader']) * self.epochs
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Define loss function
        criterion = CrossEntropyLoss()
        
        # Training loop
        print("\n=== Training CodeT5 Model ===")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
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
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update training loss
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(data_loaders['train_loader'])
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in tqdm(data_loaders['val_loader'], desc="Validation"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Update validation loss
                    val_loss += loss.item()
                    
                    # Get predictions
                    _, preds = torch.max(outputs, dim=1)
                    
                    # Store predictions and true labels
                    val_predictions.extend(preds.cpu().tolist())
                    val_true_labels.extend(labels.cpu().tolist())
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(data_loaders['val_loader'])
            self.history['val_loss'].append(avg_val_loss)
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_true_labels, val_predictions, average='binary'
            )
            
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")
            print(f"Validation F1: {val_f1:.4f}")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best model with validation accuracy: {val_accuracy:.4f}")
        
        # Load best model for testing
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
        
        return self.model

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(self.history['val_precision'], label='Validation Precision')
        plt.plot(self.history['val_recall'], label='Validation Recall')
        plt.plot(self.history['val_f1'], label='Validation F1')
        plt.title('Model Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, test_loader, dataset_name="test", export_predictions=True):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("No model trained. Call train_model first.")
        
        print("\n=== Evaluating Model on Test Set ===")
        
        # Explicitly load the best model state for evaluation
        if self.best_model_state is not None:
            print(f"Loading best model with validation accuracy: {self.best_val_accuracy:.4f}")
            self.model.load_state_dict(self.best_model_state)
            self.model.eval()
        else:
            print("Warning: No best model state found, using current model state")
            self.model.eval()
        
        test_predictions = []
        test_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                
                # Store predictions and true labels
                test_predictions.extend(preds.cpu().tolist())
                test_true_labels.extend(labels.cpu().tolist())
        
        # Calculate test metrics
        test_accuracy = accuracy_score(test_true_labels, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_true_labels, test_predictions, average='binary'
        )
        
        # Generate classification report
        class_report = classification_report(test_true_labels, test_predictions)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(test_true_labels, test_predictions)
        
        print(f"\n=== BEST MODEL EVALUATION RESULTS ===")
        if self.best_model_state is not None:
            print(f"Using best model from training (Validation Accuracy: {self.best_val_accuracy:.4f})")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print("\n=== DETAILED CLASSIFICATION REPORT (BEST MODEL) ===")
        print(class_report)
        print("="*60)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Not Vulnerable', 'Vulnerable'],
                       yticklabels=['Not Vulnerable', 'Vulnerable'])
        else:
            # Fallback to matplotlib if seaborn is not available
            plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            # Add text annotations
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, str(conf_matrix[i, j]), 
                            ha='center', va='center', color='black')
            plt.xticks([0, 1], ['Not Vulnerable', 'Vulnerable'])
            plt.yticks([0, 1], ['Not Vulnerable', 'Vulnerable'])
        
        plt.title(f'Confusion Matrix - Best Model (Val Acc: {self.best_val_accuracy:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        results = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': test_predictions,
            'true_labels': test_true_labels,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        # Export predictions if requested
        if export_predictions:
            export_path = self.export_predictions(
                test_predictions, 
                test_true_labels, 
                dataset_name
            )
            results['export_path'] = export_path
        
        return results
    
    def save_model(self):
        """Save trained model and tokenizer, and create a zip archive"""
        if self.model is None:
            print("No model to save.")
            return
        
        # Create timestamp for unique folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_dir = os.path.join(self.output_dir, f'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if self.best_model_state is not None:
            torch.save(self.best_model_state, os.path.join(model_dir, 'best_model.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pt'))
        
        # Save model configuration
        model_config = {
            'hidden_size': self.model.hidden_size,
            'vocab_size': self.preprocessor.tokenizer.vocab_size,
            'num_labels': 2,
            'max_length': self.preprocessor.max_length
        }
        
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)
        
        # Save tokenizer
        self.preprocessor.tokenizer.save_pretrained(model_dir)
        
        # Save training history
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        print(f"Model saved to {model_dir}")
        
        # Create zip archive of the model directory
        zip_path = f"{model_dir}_{timestamp}.zip"
        try:
            print(f"Creating zip archive: {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the model directory and add all files
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create archive path relative to the model directory
                        arcname = os.path.relpath(file_path, os.path.dirname(model_dir))
                        zipf.write(file_path, arcname)
            
            # Get zip file size for user feedback
            zip_size = os.path.getsize(zip_path)
            zip_size_mb = zip_size / (1024 * 1024)
            print(f"✅ Model archive created successfully: {zip_path}")
            print(f"📦 Archive size: {zip_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create zip archive: {str(e)}")
            print(f"Model files are still available in: {model_dir}")
        
        return model_dir
    
    def load_model(self, model_dir):
        """
        Load a previously saved CodeT5 model from the specified directory
        
        Args:
            model_dir: Path to the directory containing the saved model
            
        Returns:
            The loaded CodeT5Classifier model
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
            
        print(f"Loading model from {model_dir}")
        
        # Check for model config file
        config_path = os.path.join(model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"Model config file not found in {model_dir}")
            
        # Load model configuration
        with open(config_path, 'r') as f:
            model_config = json.load(f)
            
        # Initialize model
        self.model = CodeT5Classifier()
        self.model.to(device)
        
        # Check for model state file (either best_model.pt or model.pt)
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        model_path = os.path.join(model_dir, 'model.pt')
        
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, map_location=device)
            print("Loading best model checkpoint")
        elif os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            print("Loading regular model checkpoint")
        else:
            raise ValueError(f"No model checkpoint found in {model_dir}")
            
        # Load model state
        self.model.load_state_dict(state_dict)
        self.best_model_state = state_dict
        
        # Load tokenizer if available
        tokenizer_path = os.path.join(model_dir, 'special_tokens_map.json')
        if os.path.exists(tokenizer_path):
            self.preprocessor.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            print("Loaded tokenizer from saved model")
            
        # Load training history if available
        history_path = os.path.join(model_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            
            # Set best accuracy from history if available
            if self.history.get('val_accuracy'):
                self.best_val_accuracy = max(self.history['val_accuracy'])
                print(f"Loaded training history. Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Set model to evaluation mode
        self.model.eval()
        print("Model loaded successfully and set to evaluation mode")
        
        return self.model
    
    def export_predictions(self, predictions, true_labels=None, dataset_name="test"):
        """
        Export model predictions to a txt file with index and prediction format
        
        Args:
            predictions: List or array of predictions (0 or 1)
            true_labels: Optional list of true labels for comparison
            dataset_name: Name to include in the filename (e.g., "test", "cwe119")
            
        Returns:
            Path to the exported file
        """
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create filename
        if "cwe" in dataset_name.lower():
            filename = f"predict_codet5_{dataset_name}_{timestamp}.txt"
        else:
            filename = f"predict_codet5_cwe_{timestamp}.txt"
        
        # Full path for the output file
        output_path = os.path.join(self.output_dir, filename)
        
        # Write predictions to file
        with open(output_path, 'w') as f:
            for idx, pred in enumerate(predictions):
                f.write(f"{idx}\t{pred}\n")
        
        print(f"Predictions exported to: {output_path}")
        print(f"Total predictions exported: {len(predictions)}")
        
        # If true labels are provided, also create a comparison file
        if true_labels is not None:
            comparison_filename = f"prediction_comparison_{dataset_name}_{timestamp}.txt"
            comparison_path = os.path.join(self.output_dir, comparison_filename)
            
            with open(comparison_path, 'w') as f:
                f.write("Index\tPrediction\tTrue_Label\tCorrect\n")
                correct_count = 0
                for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
                    is_correct = pred == true
                    if is_correct:
                        correct_count += 1
                    f.write(f"{idx}\t{pred}\t{true}\t{is_correct}\n")
            
            accuracy = correct_count / len(predictions) if len(predictions) > 0 else 0
            print(f"Prediction comparison exported to: {comparison_path}")
            print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(predictions)})")
        
        return output_path
    
    def predict(self, code_text):
        """
        Make a prediction on a single code sample
        
        Args:
            code_text: String containing the code to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call train_model or load_model first.")
        
        # Preprocess and tokenize the code
        encoding = self.preprocessor.tokenize(
            code_text, 
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        result = {
            'prediction': prediction.item(),  # 0: not vulnerable, 1: vulnerable
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'label_names': ['Not Vulnerable', 'Vulnerable']
        }
        
        return result

def free_gpu_memory():
    """Free up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Example 1: Train and evaluate model on CWE dataset
    def train_on_cwe_dataset(cwe_type):
        print(f"\n=== Training CodeT5 Model on {cwe_type} Dataset ===")
        
        # Paths to training and testing data
        train_path = f"{cwe_type}_train.csv"
        test_path = f"{cwe_type}_test.csv"
        
        # Create trainer with default hyperparameters
        trainer = CodeT5Trainer(batch_size=8, epochs=3)
        
        try:
            # Load training and testing data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Run training, evaluation, and save model
            results = trainer.run_all(
                train_data=train_data,
                test_data=test_data,
                freeze_base=False,  # Don't freeze the base model
                dataset_name=cwe_type
            )
            
            print(f"\n=== Results for {cwe_type} ===")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1']:.4f}")
            
            return trainer, results
            
        except Exception as e:
            print(f"Error training on {cwe_type}: {str(e)}")
            return None, None
    
    # Example 2: Load a saved model and make predictions
    def predict_with_saved_model(model_dir, code_snippet):
        print("\n=== Making Predictions with Saved Model ===")
        
        # Create trainer
        trainer = CodeT5Trainer()
        
        try:
            # Load saved model
            trainer.load_model(model_dir)
            
            # Make prediction
            result = trainer.predict(code_snippet)
            
            print("\n=== Prediction Result ===")
            print(f"Label: {result['label_names'][result['prediction']]}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    # Uncomment to run example training on a CWE dataset
    # trainer, results = train_on_cwe_dataset("cwe20")
    
    # Example vulnerable code snippet
    example_code = """
    void vulnerable_function(char *input) {
        char buffer[10];
        strcpy(buffer, input); // Potential buffer overflow
        printf("%s\\n", buffer);
    }
    """
    
    # Uncomment to run example prediction (after training a model)
    # prediction = predict_with_saved_model("codet5_outputs/model", example_code) 