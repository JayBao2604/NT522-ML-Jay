import numpy as np
import pandas as pd
import torch
import os
import re
import random
import glob
import json
import math
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import sys
try:
    import seaborn as sns
except ImportError:
    print("Warning: Seaborn not installed. Some visualizations may not work.")

# Import CodeBERT trainer and utility functions
from train_codebert_model import CodeBERTTrainer, free_gpu_memory
from fga_selection import centriod_init, calcaulate_weight, select, update_global_pop


class AdversarialLearning:
    """
    Adversarial Learning class implementing Fuzzy Genetic Algorithm for
    optimizing adversarial samples against a CodeBERT vulnerability detection model.
    """
    
    def __init__(self, attack_pool_path="attack_pool.csv", model_path=None, 
                 pop_size=20, clusters=3, max_generations=50, decay_rate=1.5, 
                 alpha=2.0, penalty=0.01, verbose=1):
        """
        Initialize the Adversarial Learning with FGA
        
        Args:
            attack_pool_path: Path to the attack pool CSV
            model_path: Path to the trained CodeBERT model (optional)
            pop_size: Population size for genetic algorithm
            clusters: Number of fuzzy clusters
            max_generations: Maximum number of generations
            decay_rate: Decay rate for fuzzy clustering
            alpha: Fuzziness factor
            penalty: Penalty factor for code snippet length
            verbose: Verbosity level
        """
        self.attack_pool_path = attack_pool_path
        self.model_path = model_path
        self.pop_size = pop_size
        self.clusters = clusters
        self.max_generations = max_generations
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.penalty = penalty
        self.verbose = verbose
        
        # Load attack pool
        self.attack_pool = self._load_attack_pool()
        
        # Initialize model trainer and model
        self.trainer = None
        self.model = None
        if model_path:
            self._load_model(model_path)
        
        # Initialize population and centroids
        self.population = {}
        self.centroids = None
        
        # Add storage for loaded predictions
        self.original_predictions = None
        self.prediction_file_path = None
        
    def _load_attack_pool(self):
        """Load the attack pool CSV file"""
        if not os.path.exists(self.attack_pool_path):
            raise FileNotFoundError(f"Attack pool file not found: {self.attack_pool_path}")
        
        attack_pool = pd.read_csv(self.attack_pool_path)
        
        if self.verbose:
            print(f"\n=== ATTACK POOL LOADING ===")
            print(f"Raw attack pool shape: {attack_pool.shape}")
            print(f"Available columns: {list(attack_pool.columns)}")
        
        # Handle different attack pool formats - only expect adversarial code
        if 'adversarial_code' in attack_pool.columns:
            # Standard format: adversarial_code column
            if self.verbose:
                print(f"Detected attack pool format with 'adversarial_code' column")
            attack_pool_standardized = attack_pool[['adversarial_code']].copy()
            
        else:
            # Try to auto-detect format based on available columns
            available_columns = list(attack_pool.columns)
            if self.verbose:
                print(f"Available columns in attack pool: {available_columns}")
            
            # If there's only one column, assume it contains adversarial code
            if len(available_columns) == 1:
                adversarial_column = available_columns[0]
                if self.verbose:
                    print(f"Using single column '{adversarial_column}' as adversarial code")
                
                attack_pool_standardized = pd.DataFrame({
                    'adversarial_code': attack_pool[adversarial_column].values
                })
            else:
                raise ValueError(f"Attack pool format not recognized. Expected 'adversarial_code' column. Found columns: {available_columns}")
        
        # Remove any rows with NaN values
        initial_size = len(attack_pool_standardized)
        attack_pool_standardized = attack_pool_standardized.dropna()
        final_size = len(attack_pool_standardized)
        
        if self.verbose:
            print(f"Attack pool processed successfully:")
            print(f"  Initial size: {initial_size}")
            print(f"  After removing NaN: {final_size}")
            print(f"  Final shape: {attack_pool_standardized.shape}")
            print(f"Sample adversarial codes:")
            for i, code in enumerate(attack_pool_standardized['adversarial_code'].head(3)):
                print(f"  [{i+1}] {code[:100]}{'...' if len(code) > 100 else ''}")
        
        return attack_pool_standardized
    
    def _load_model(self, model_path):
        """Load the CodeBERT model"""
        try:
            # Initialize the trainer
            self.trainer = CodeBERTTrainer()
            
            # Load model from saved path
            if os.path.exists(model_path):
                self.model = self.trainer.load_model(model_path)
                
                # CRITICAL FIX: Ensure model is in evaluation mode
                if self.model is not None:
                    self.model.eval()
                    # Also ensure trainer's model is in eval mode
                    if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                        self.trainer.model.eval()
                
                if self.verbose:
                    print(f"Successfully loaded model from {model_path}")
                    print(f"Model is in eval mode: {not self.model.training}")
                    
                    # Test model prediction to verify it's working
                    test_code = "void test() { char buf[10]; }"
                    try:
                        test_pred = self.trainer.predict(test_code)
                        print(f"Model test prediction: {test_pred}")
                    except Exception as e:
                        print(f"Warning: Model test prediction failed: {str(e)}")
            else:
                self.model = None
                if self.verbose:
                    print(f"Model path {model_path} not found. Will train a new model when needed.")
        except Exception as e:
            self.model = None
            print(f"Error loading model: {str(e)}")
            print(f"Model path attempted: {model_path}")
            if os.path.exists(model_path):
                print(f"Path exists but model loading failed")
                # List files in model directory for debugging
                if os.path.isdir(model_path):
                    print(f"Files in model directory: {os.listdir(model_path)}")
            else:
                print(f"Model path does not exist")
    
    def initialize_population(self):
        """Initialize the population with random adversarial code snippets"""
        if self.verbose:
            print(f"\n=== POPULATION INITIALIZATION ===")
            print(f"Attack pool size: {len(self.attack_pool)}")
            print(f"Requested population size: {self.pop_size}")
        
        # Sample from attack pool to create initial population
        if len(self.attack_pool) < self.pop_size:
            # If attack pool is smaller than pop_size, duplicate some samples
            indices = np.random.choice(len(self.attack_pool), self.pop_size, replace=True)
            if self.verbose:
                print(f"Attack pool smaller than population size - sampling with replacement")
        else:
            # Sample without replacement
            indices = np.random.choice(len(self.attack_pool), self.pop_size, replace=False)
            if self.verbose:
                print(f"Attack pool larger than population size - sampling without replacement")
        
        # Initialize population dictionary with fitness scores set to 0
        self.population = {}
        for idx in indices:
            adv_code = self.attack_pool.iloc[idx]['adversarial_code']
            self.population[adv_code] = 0  # Initial fitness score
        
        # Initialize centroids from uniform distribution
        min_distance = 1.0 / (self.clusters * 2)  # Ensure centroids are reasonably spaced
        self.centroids = centriod_init(self.clusters, min_distance)
        
        if self.verbose:
            print(f"Population successfully initialized:")
            print(f"  Population size: {len(self.population)}")
            print(f"  Unique adversarial codes: {len(set(self.population.keys()))}")
            print(f"  Clusters: {self.clusters}")
            print(f"  Initial centroids: {self.centroids}")
            
            # Show a few sample adversarial codes from the population
            print(f"Sample population codes:")
            for i, code in enumerate(list(self.population.keys())[:3]):
                print(f"  [{i+1}] {code[:80]}{'...' if len(code) > 80 else ''}")
        
        return self.population, self.centroids
    
    def calculate_fitness(self, original_df, adversarial_code, model=None, return_attack_rate=False):
        """
        Calculate fitness score for an adversarial code snippet
        
        Args:
            original_df: DataFrame containing original code
            adversarial_code: Adversarial code snippet
            model: Trained model (if None, will use loaded predictions from txt)
            
        Returns:
            Fitness score based on attack success rate and snippet length
        """
        # Create a copy of the original dataframe
        adv_df = original_df.copy()
        
        # Apply the adversarial code to each sample
        # Only add adversarial code to samples labeled as vulnerable (label=1)
        vulnerable_samples = adv_df['label'] == 1
        num_vulnerable = vulnerable_samples.sum()
        
        if num_vulnerable == 0:
            # If no vulnerable samples, make some samples vulnerable for testing
            if self.verbose >= 2:
                print("No vulnerable samples found. Creating synthetic vulnerable samples.")
            # Mark a portion of samples as vulnerable for testing
            sample_indices = np.random.choice(len(adv_df), max(1, len(adv_df) // 4), replace=False)
            adv_df.loc[sample_indices, 'label'] = 1
            vulnerable_samples = adv_df['label'] == 1
            num_vulnerable = vulnerable_samples.sum()
        
        # ENHANCED DIAGNOSTICS: Show dataset composition
        if self.verbose >= 2:
            total_samples = len(original_df)
            vulnerable_labeled = (original_df['label'] == 1).sum()
            benign_labeled = (original_df['label'] == 0).sum()
            print(f"\n=== DATASET COMPOSITION ===")
            print(f"Total samples: {total_samples}")
            print(f"Labeled as vulnerable: {vulnerable_labeled}")
            print(f"Labeled as benign: {benign_labeled}")
            print(f"Applying adversarial code to {num_vulnerable} vulnerable samples")
        
        # Use loaded predictions if available, otherwise use model
        if self.original_predictions is not None:
            if self.verbose >= 2:
                print("Using loaded predictions from txt file")
            
            # Ensure predictions match the dataset size
            if len(self.original_predictions) != len(original_df):
                print(f"Warning: Prediction length ({len(self.original_predictions)}) doesn't match dataset length ({len(original_df)})")
                # Try to align predictions with dataset
                if len(self.original_predictions) > len(original_df):
                    original_predictions = self.original_predictions[:len(original_df)]
                else:
                    # Pad with zeros if needed
                    original_predictions = np.pad(self.original_predictions, 
                                                (0, len(original_df) - len(self.original_predictions)), 
                                                constant_values=0)
            else:
                original_predictions = self.original_predictions.copy()
        else:
            # Fall back to model predictions if no txt file is loaded
            if self.verbose >= 2:
                print("No loaded predictions found, using model to predict")
            
            # Set verbosity based on self.verbose level
            if self.verbose <= 1:
                # Temporarily reduce print output
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
            
            # Function to get predictions using the model directly
            def get_predictions(df):
                predictions = []
                try:
                    # CRITICAL FIX: Ensure model is in evaluation mode before predictions
                    if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                        self.trainer.model.eval()
                    
                    for _, row in df.iterrows():
                        code = row['functionSource']
                        # Try the trainer's predict method with error handling
                        try:
                            pred = self.trainer.predict(code)
                            if isinstance(pred, dict) and 'prediction' in pred:
                                predictions.append(pred['prediction'])
                            else:
                                # If the predict method returns an unexpected format,
                                # just use a default prediction of non-vulnerable
                                if self.verbose >= 2:
                                    print(f"Predict returned unexpected format: {pred}")
                                predictions.append(0)  # Default to non-vulnerable
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"Error in prediction for sample: {str(e)}")
                                print(f"Code length: {len(code)}")
                            # Default to predicting as non-vulnerable (0) if there's an error
                            predictions.append(0)
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"Error in get_predictions: {str(e)}")
                    # Return all zeros if there's a major error
                    predictions = [0] * len(df)
                return np.array(predictions)
            
            # Get predictions on original data
            original_predictions = get_predictions(original_df)
            
            if self.verbose <= 1:
                # Restore print output
                sys.stdout.close()
                sys.stdout = old_stdout

        # Count initially vulnerable samples that were correctly predicted as vulnerable
        vulnerable_indices = np.where(vulnerable_samples)[0]
        correctly_identified_vulnerabilities = sum(1 for i in vulnerable_indices 
                                                 if original_predictions[i] == 1)

        # ENHANCED DIAGNOSTICS: Show model performance breakdown
        if self.verbose >= 2:
            print(f"\n=== MODEL PERFORMANCE BREAKDOWN ===")
            
            # Count predictions by true label
            true_vulnerable_indices = np.where(original_df['label'] == 1)[0]
            true_benign_indices = np.where(original_df['label'] == 0)[0]
            
            # For vulnerable samples
            vuln_pred_as_vuln = sum(1 for i in true_vulnerable_indices if original_predictions[i] == 1)
            vuln_pred_as_benign = sum(1 for i in true_vulnerable_indices if original_predictions[i] == 0)
            
            # For benign samples  
            benign_pred_as_vuln = sum(1 for i in true_benign_indices if original_predictions[i] == 1)
            benign_pred_as_benign = sum(1 for i in true_benign_indices if original_predictions[i] == 0)
            
            print(f"Vulnerable samples (label=1): {len(true_vulnerable_indices)} total")
            print(f"  → Predicted as vulnerable: {vuln_pred_as_vuln}")
            print(f"  → Predicted as benign: {vuln_pred_as_benign}")
            print(f"Benign samples (label=0): {len(true_benign_indices)} total")
            print(f"  → Predicted as vulnerable: {benign_pred_as_vuln}")
            print(f"  → Predicted as benign: {benign_pred_as_benign}")
            
            model_accuracy = (vuln_pred_as_vuln + benign_pred_as_benign) / len(original_df)
            vulnerable_recall = vuln_pred_as_vuln / len(true_vulnerable_indices) if len(true_vulnerable_indices) > 0 else 0
            
            print(f"Model accuracy: {model_accuracy:.4f}")
            print(f"Vulnerable recall: {vulnerable_recall:.4f}")
            print(f"Samples available for attack: {correctly_identified_vulnerabilities}")
        
        if correctly_identified_vulnerabilities == 0:
            if self.verbose >= 2:
                print("No vulnerabilities correctly identified by model. Attack cannot succeed.")
            # Return zero fitness since we can't measure attack success
            if return_attack_rate:
                return 0.0, 0.0
            else:
                return 0.0
        
        # Insert adversarial code into vulnerable samples using improved strategies
        for idx in adv_df.index[vulnerable_samples]:
            # Skip samples not originally predicted as vulnerable
            if original_predictions[idx] != 1:
                continue
                
            # Insert adversarial code using multiple improved strategies
            orig_code = adv_df.loc[idx, 'functionSource']
            
            # Strategy 1: Insert at function level (after function declaration)
            # Strategy 2: Insert at variable declaration level
            # Strategy 3: Insert at loop/conditional level
            # Strategy 4: Insert at critical computation points
            
            code_lines = orig_code.split('\n')
            code_len = len(code_lines)
            
            # More sophisticated insertion strategy
            # Look for function declarations, variable declarations, loops, etc.
            enhanced_code_lines = code_lines.copy()
            insertions_made = 0
            
            for i, line in enumerate(code_lines):
                line_stripped = line.strip().lower()
                
                # Insert after function declarations
                if (any(keyword in line_stripped for keyword in ['void ', 'int ', 'char ', 'function ', 'def ']) and 
                    ('(' in line and ')' in line and '{' in line) and insertions_made < 2):
                    enhanced_code_lines.insert(i + 1 + insertions_made, f"    {adversarial_code}")
                    insertions_made += 1
                
                # Insert before variable declarations involving user input
                elif (any(keyword in line_stripped for keyword in ['scanf', 'gets', 'input', 'read']) and 
                      insertions_made < 3):
                    enhanced_code_lines.insert(i + insertions_made, f"    {adversarial_code}")
                    insertions_made += 1
                
                # Insert in loop bodies
                elif (any(keyword in line_stripped for keyword in ['for ', 'while ', 'do ']) and 
                      insertions_made < 2):
                    enhanced_code_lines.insert(i + 1 + insertions_made, f"        {adversarial_code}")
                    insertions_made += 1
            
            # If no strategic insertions were made, fall back to fixed positions
            if insertions_made == 0:
                # Use more aggressive insertion at multiple fixed positions
                positions = [
                    min(2, code_len - 1),     # Near the beginning
                    code_len // 2,            # Middle
                    max(1, code_len - 2)      # Near the end
                ]
                
                for i, pos in enumerate(positions):
                    enhanced_code_lines.insert(pos + i, f"    {adversarial_code}")
            
            # Apply the modified code
            adv_df.loc[idx, 'functionSource'] = '\n'.join(enhanced_code_lines)
        
        # Get adversarial predictions
        if self.original_predictions is not None and hasattr(self, 'trainer') and self.trainer is not None:
            # Use model to predict adversarial samples since we need new predictions
            if self.verbose >= 2:
                print("Using model to predict adversarial samples")
            
            # Set verbosity based on self.verbose level
            if self.verbose <= 1:
                # Temporarily reduce print output
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
            
            # Function to get predictions using the model directly
            def get_adversarial_predictions(df):
                predictions = []
                try:
                    # CRITICAL FIX: Ensure model is in evaluation mode before predictions
                    if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                        self.trainer.model.eval()
                    
                    for _, row in df.iterrows():
                        code = row['functionSource']
                        # Try the trainer's predict method with error handling
                        try:
                            pred = self.trainer.predict(code)
                            if isinstance(pred, dict) and 'prediction' in pred:
                                predictions.append(pred['prediction'])
                            else:
                                predictions.append(0)  # Default to non-vulnerable
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"Error in adversarial prediction for sample: {str(e)}")
                            predictions.append(0)
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"Error in get_adversarial_predictions: {str(e)}")
                    predictions = [0] * len(df)
                return np.array(predictions)
            
            # Get predictions on adversarial data
            adversarial_predictions = get_adversarial_predictions(adv_df)
            
            if self.verbose <= 1:
                # Restore print output
                sys.stdout.close()
                sys.stdout = old_stdout
        else:
            # If no model available, assume adversarial attack fails
            adversarial_predictions = original_predictions.copy()
        
        # FIXED: Calculate attack success rate properly
        # Focus only on vulnerable samples that were correctly identified initially
        initially_vulnerable_and_detected = []
        for i in vulnerable_indices:
            if original_predictions[i] == 1:  # Was correctly identified as vulnerable
                initially_vulnerable_and_detected.append(i)
        
        # Count successful attacks (vulnerable samples that became non-vulnerable)
        successful_attacks = 0
        total_prediction_changes = 0
        zero_to_one = 0  # Non-vulnerable to vulnerable
        one_to_zero = 0  # Vulnerable to non-vulnerable (this is what we want)
        
        for i in initially_vulnerable_and_detected:
            if adversarial_predictions[i] != original_predictions[i]:
                total_prediction_changes += 1
                if original_predictions[i] == 1 and adversarial_predictions[i] == 0:
                    successful_attacks += 1
                    one_to_zero += 1
        
        # Also check all samples for any changes (for debugging)
        for i in range(len(original_predictions)):
            if original_predictions[i] != adversarial_predictions[i]:
                if original_predictions[i] == 0 and adversarial_predictions[i] == 1:
                    zero_to_one += 1
        
        # Calculate attack success rate based on initially vulnerable and detected samples only
        if len(initially_vulnerable_and_detected) > 0:
            attack_success_rate = successful_attacks / len(initially_vulnerable_and_detected)
        else:
            attack_success_rate = 0.0
        
        if self.verbose >= 2:
            print(f"Attack success rate: {attack_success_rate:.4f} ({successful_attacks}/{len(initially_vulnerable_and_detected)} samples changed prediction)")
            print(f"  - 0→1 changes: {zero_to_one}/{len(original_predictions)} ({zero_to_one/len(original_predictions):.4f})")
            print(f"  - 1→0 changes: {one_to_zero}/{len(original_predictions)} ({one_to_zero/len(original_predictions):.4f})")
        
        # Calculate penalty for code snippet length
        snippet_length = len(adversarial_code.splitlines())
        length_penalty = self.penalty * snippet_length
        
        # Calculate fitness score - heavily weight attack success
        # Use a more aggressive fitness function that rewards high attack success rates
        if attack_success_rate > 0.8:  # Bonus for very high success rates
            fitness_score = attack_success_rate + 0.2 - length_penalty
        elif attack_success_rate > 0.5:  # Bonus for moderate success rates
            fitness_score = attack_success_rate + 0.1 - length_penalty
        else:
            fitness_score = attack_success_rate - length_penalty
        
        if self.verbose >= 2:
            print(f"Adversarial snippet length: {snippet_length}")
            print(f"Length penalty: {length_penalty:.4f}")
            print(f"Fitness score: {fitness_score:.4f}")
        
        if return_attack_rate:
            return fitness_score, attack_success_rate
        else:
            return fitness_score
    
    def perform_fuzzy_clustering(self):
        """Perform fuzzy clustering on the population"""
        # Get fitness scores as array
        keys = list(self.population.keys())
        scores = np.array([self.population[k] for k in keys])
        
        # Calculate fuzzy membership weights for each sample
        membership_weights = {}
        for key, score in zip(keys, scores):
            weight = calcaulate_weight(score, self.centroids)
            membership_weights[key] = weight
        
        # Update centroids based on weighted scores
        new_centroids = np.zeros_like(self.centroids)
        for k in range(len(self.centroids)):
            numerator = 0
            denominator = 0
            
            for key, score in zip(keys, scores):
                weight = membership_weights[key][k]
                weight_alpha = weight ** self.alpha
                numerator += weight_alpha * score
                denominator += weight_alpha
            
            new_centroids[k] = numerator / denominator if denominator > 0 else self.centroids[k]
        
        # Check for convergence
        centroid_change = np.sum(np.abs(new_centroids - self.centroids))
        self.centroids = new_centroids
        
        if self.verbose >= 2:
            print(f"Updated centroids: {self.centroids}")
            print(f"Centroid change: {centroid_change:.6f}")
        
        return membership_weights, centroid_change
    
    def select_clusters(self):
        """Select top 2 clusters based on centroid magnitude"""
        # Sort centroids by magnitude (fitness score)
        sorted_indices = np.argsort(self.centroids)[::-1]
        top_clusters = sorted_indices[:2]  # Select top 2 clusters
        
        if self.verbose >= 2:
            print(f"Selected top clusters: {top_clusters} with centroids {self.centroids[top_clusters]}")
        
        return top_clusters
    
    def perform_crossover(self, membership_weights, top_clusters):
        """Perform crossover operation to create offspring"""
        keys = list(self.population.keys())
        
        # Select parents from top clusters
        parents = []
        for _ in range(self.pop_size // 2):  # Create pop_size/2 offspring
            # Use the original select function from fga_selection.py with error handling
            try:
                # Select parent from first top cluster
                parent1 = select(self.population, self.centroids[top_clusters[0]], 
                                self.centroids, self.decay_rate)
                
                # Select parent from second top cluster  
                parent2 = select(self.population, self.centroids[top_clusters[1]], 
                                self.centroids, self.decay_rate)
                
                parents.append((parent1, parent2))
            except (ZeroDivisionError, ValueError, np.linalg.LinAlgError, OverflowError) as e:
                # Handle numerical issues from original select function
                if self.verbose >= 2:
                    print(f"Numerical error in parent selection: {str(e)}")
                    print("Falling back to random selection")
                
                parent1 = random.choice(keys)
                parent2 = random.choice(keys)
                parents.append((parent1, parent2))
            except Exception as e:
                # Fallback to random selection if there's any other issue
                if self.verbose >= 2:
                    print(f"Error in parent selection: {str(e)}")
                    print("Falling back to random selection")
                
                parent1 = random.choice(keys)
                parent2 = random.choice(keys)
                parents.append((parent1, parent2))
        
        # Create offspring through improved crossover strategies
        offspring = []
        for parent1, parent2 in parents:
            # Enhanced crossover: create multiple offspring variations per parent pair
            for variation in range(2):  # Create 2 variations per parent pair
                p1_lines = parent1.split('\n')
                p2_lines = parent2.split('\n')
                
                # Intelligent crossover that preserves vulnerability patterns
                if len(p1_lines) <= 1 or len(p2_lines) <= 1:
                    # For very short snippets, combine them strategically
                    if variation == 0:
                        child = parent1 + '; ' + parent2  # Combine on same line
                    else:
                        child = parent1 + '\n' + parent2  # Combine on separate lines
                else:
                    # Multiple sophisticated crossover strategies
                    strategy = random.choice(['semantic_mix', 'vulnerability_focused', 'pattern_preservation', 'obfuscation_mix'])
                    
                    if strategy == 'semantic_mix':
                        # Mix based on semantic patterns (vulnerabilities vs normal code)
                        vuln_keywords = ['malloc', 'free', 'strcpy', 'gets', 'sprintf', 'system', 'exec']
                        
                        # Separate vulnerable and normal lines
                        p1_vuln = [line for line in p1_lines if any(kw in line.lower() for kw in vuln_keywords)]
                        p1_normal = [line for line in p1_lines if not any(kw in line.lower() for kw in vuln_keywords)]
                        p2_vuln = [line for line in p2_lines if any(kw in line.lower() for kw in vuln_keywords)]
                        p2_normal = [line for line in p2_lines if not any(kw in line.lower() for kw in vuln_keywords)]
                        
                        # Combine vulnerable parts from both parents with normal parts
                        child_lines = []
                        if p1_vuln: child_lines.extend(p1_vuln[:2])  # Take first 2 vuln lines from p1
                        if p2_normal: child_lines.extend(p2_normal[:1])  # Mix with normal from p2
                        if p2_vuln: child_lines.extend(p2_vuln[:2])  # Take vuln lines from p2
                        if p1_normal: child_lines.extend(p1_normal[:1])  # Mix with normal from p1
                    
                    elif strategy == 'vulnerability_focused':
                        # Focus on combining different types of vulnerabilities
                        # Take the most dangerous-looking lines from each parent
                        danger_keywords = ['overflow', 'injection', 'format', 'buffer', 'memory', 'null', 'free', 'alloc']
                        
                        p1_danger = [line for line in p1_lines if any(kw in line.lower() for kw in danger_keywords)]
                        p2_danger = [line for line in p2_lines if any(kw in line.lower() for kw in danger_keywords)]
                        
                        child_lines = []
                        # Interleave dangerous patterns
                        max_danger = max(len(p1_danger), len(p2_danger))
                        for i in range(max_danger):
                            if i < len(p1_danger):
                                child_lines.append(p1_danger[i])
                            if i < len(p2_danger):
                                child_lines.append(p2_danger[i])
                        
                        # Fill in with remaining lines if needed
                        if not child_lines:
                            child_lines = p1_lines[:len(p1_lines)//2] + p2_lines[len(p2_lines)//2:]
                    
                    elif strategy == 'pattern_preservation':
                        # Preserve important patterns while mixing
                        # Look for function calls, variable declarations, etc.
                        p1_funcs = [line for line in p1_lines if '(' in line and ')' in line]
                        p1_vars = [line for line in p1_lines if any(typ in line.lower() for typ in ['char', 'int', 'void', 'size_t'])]
                        p2_funcs = [line for line in p2_lines if '(' in line and ')' in line]
                        p2_vars = [line for line in p2_lines if any(typ in line.lower() for typ in ['char', 'int', 'void', 'size_t'])]
                        
                        child_lines = []
                        # Combine variable declarations and function calls strategically
                        if p1_vars: child_lines.extend(p1_vars[:2])
                        if p2_funcs: child_lines.extend(p2_funcs[:2])
                        if p2_vars: child_lines.extend(p2_vars[:1])
                        if p1_funcs: child_lines.extend(p1_funcs[:2])
                    
                    else:  # obfuscation_mix
                        # Create obfuscated combinations
                        # Take parts from each parent and add obfuscating comments
                        p1_half = len(p1_lines) // 2
                        p2_half = len(p2_lines) // 2
                        
                        child_lines = []
                        child_lines.extend(p1_lines[:p1_half])
                        child_lines.append("// Security check passed")  # Obfuscating comment
                        child_lines.extend(p2_lines[p2_half:])
                        if random.random() < 0.5:
                            child_lines.append("// Code reviewed and approved")  # More obfuscation
                
                child = '\n'.join(child_lines) if 'child_lines' in locals() else parent1
                
                # Ensure the child is not empty or too short
                if len(child.strip()) < 5:
                    child = parent1 if len(parent1) > len(parent2) else parent2  # Use longer parent
                
                # Add mutation to create more diversity
                if random.random() < 0.3:  # 30% mutation rate
                    child = self._mutate_adversarial_code(child)
                    
                offspring.append(child)
        
        # Ensure we don't exceed population size
        offspring = offspring[:self.pop_size//2]
        
        if self.verbose >= 2:
            print(f"Created {len(offspring)} offspring through enhanced crossover")
        
        return offspring
    
    def _mutate_adversarial_code(self, code):
        """Apply mutation to adversarial code to increase diversity"""
        lines = code.split('\n')
        
        mutation_types = ['add_comment', 'modify_variable', 'add_vulnerability', 'obfuscate']
        mutation = random.choice(mutation_types)
        
        if mutation == 'add_comment':
            # Add misleading comments
            comments = [
                "// Bounds checked above",
                "// Input sanitized",
                "// Memory properly allocated",
                "// Safe operation confirmed",
                "// Validated by security team"
            ]
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, random.choice(comments))
            
        elif mutation == 'modify_variable':
            # Modify variable names to be more misleading
            replacements = {
                'buffer': 'safe_buffer',
                'input': 'validated_input',
                'ptr': 'safe_ptr',
                'query': 'sanitized_query'
            }
            for i, line in enumerate(lines):
                for old, new in replacements.items():
                    if old in line:
                        lines[i] = line.replace(old, new)
                        break
        
        elif mutation == 'add_vulnerability':
            # Add additional vulnerability patterns
            vuln_patterns = [
                "strcpy(temp, user_data); // Fast copy",
                "system(command); // Execute utility",
                "free(ptr); // Cleanup memory",
                "sprintf(msg, format, data); // Format message"
            ]
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, random.choice(vuln_patterns))
            
        else:  # obfuscate
            # Add obfuscating code
            obfuscations = [
                "if(1) { // Always true condition",
                "int dummy = 0; // Temporary variable",
                "/* Multi-line comment for clarity */",
                "#ifdef DEBUG",
                "#endif"
            ]
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, random.choice(obfuscations))
        
        return '\n'.join(lines)
    
    def run(self, original_data_path=None, prediction_file_path=None):
        """
        Run the adversarial learning process
        
        Args:
            original_data_path: Path to original data CSV (if None, will create synthetic data)
            prediction_file_path: Path to txt file containing model predictions (optional)
            
        Returns:
            Best adversarial code snippet
        """
        print("\n===== ADVERSARIAL LEARNING DIAGNOSTICS =====")
        
        # Load predictions from txt file if provided
        if prediction_file_path:
            print(f"Loading predictions from: {prediction_file_path}")
            self.load_predictions_from_txt(prediction_file_path)
        
        # Load or create original data
        if original_data_path and os.path.exists(original_data_path):
            original_df = pd.read_csv(original_data_path)
            if 'functionSource' not in original_df.columns or 'label' not in original_df.columns:
                raise ValueError("Original data must contain 'functionSource' and 'label' columns")
            print(f"Loaded original data from {original_data_path}")
        else:
            # Create synthetic data for testing purposes
            print("No original data path provided, creating synthetic data for testing...")
            synthetic_functions = [
                "void func1() { char buf[100]; return; }",
                "int func2(char* input) { int len = strlen(input); return len; }",
                "void func3() { int* ptr = malloc(sizeof(int)); free(ptr); }",
                "char* func4(int size) { return malloc(size); }",
                "void func5(char* str) { printf(\"%s\", str); }",
                "int func6() { char buffer[256]; gets(buffer); return 0; }",
                "void func7(char* dest, char* src) { strcpy(dest, src); }",
                "int func8(char* cmd) { return system(cmd); }",
            ]
            
            # Repeat synthetic functions to match attack pool size if needed
            num_samples = max(len(self.attack_pool), 50)  # At least 50 samples
            extended_functions = []
            for i in range(num_samples):
                base_func = synthetic_functions[i % len(synthetic_functions)]
                # Add variation to make functions unique
                modified_func = base_func.replace("func", f"func_{i}")
                extended_functions.append(modified_func)
            
            original_df = pd.DataFrame({
                'functionSource': extended_functions,
                'label': np.zeros(len(extended_functions), dtype=int)  # All non-vulnerable initially
            })
            
            # Add some vulnerable examples
            vulnerable_indices = np.random.choice(len(original_df), len(original_df) // 3, replace=False)
            original_df.loc[vulnerable_indices, 'label'] = 1
            print(f"Created synthetic data with {len(original_df)} samples ({len(vulnerable_indices)} vulnerable)")
        
        # Print data composition
        vulnerable_count = (original_df['label'] == 1).sum()
        benign_count = (original_df['label'] == 0).sum()
        print(f"Data composition: {vulnerable_count} vulnerable, {benign_count} benign samples")
        
        if vulnerable_count == 0:
            print("WARNING: No vulnerable samples in dataset! Creating synthetic vulnerable examples.")
            # Create some synthetic vulnerable examples
            n_synthetic = max(10, len(original_df) // 4)
            synthetic_indices = np.random.choice(len(original_df), n_synthetic, replace=False)
            original_df.loc[synthetic_indices, 'label'] = 1
            vulnerable_count = (original_df['label'] == 1).sum()
            print(f"Added {vulnerable_count} synthetic vulnerable examples")
        
        # Initialize population and centroids
        self.initialize_population()
        
        # Create more aggressive adversarial examples for the initial population
        print("Enhancing attack pool with more aggressive adversarial examples...")
        aggressive_examples = [
            # Comments to mislead
            "// This is a harmless comment that might confuse the model",
            "/* This code was previously flagged as vulnerable */",
            "// Code reviewed by security team",
            
            # Actual vulnerable patterns but used in harmless ways
            "char buffer[100]; // Large enough buffer",
            "if(strlen(input) < sizeof(buffer)) strcpy(buffer, input);",
            "system(\"echo Hello\"); // Fixed command, not injection",
            
            # Confusing constructs
            "int *ptr = malloc(10); if(ptr != NULL) { free(ptr); ptr = NULL; }",
            "char *sql = \"SELECT * FROM users WHERE id=?\"; // Parameterized query",
            "FILE *f = fopen(filename, \"r\"); if(f != NULL) { fclose(f); }",
            
            # Code that looks harmless but has subtle issues
            "char *p = malloc(strlen(s)); strcpy(p, s); // Missing +1 for null terminator",
            "int size = n * sizeof(int); char *buf = malloc(size); // Potential integer overflow",
            "printf(\"Value: %s\", str); // Potentially format string vulnerability if str contains %",
            
            # More aggressive exploits (obvious vulnerabilities)
            "gets(buffer); // Known buffer overflow",
            "strcpy(dst, src); // No bounds checking",
            "system(user_input); // Command injection",
            "exec(user_input); // Command execution",
            "sprintf(query, \"SELECT * FROM users WHERE name='%s'\", user_input); // SQL injection",
            "free(ptr); free(ptr); // Double free"
        ]
        
        # Replace some population members with these examples
        population_keys = list(self.population.keys())
        for i in range(min(len(aggressive_examples), len(population_keys))):
            self.population[aggressive_examples[i]] = 0
            if i < len(population_keys):
                del self.population[population_keys[i]]
        
        # Initialize model if needed (only if predictions weren't loaded from txt)
        if self.original_predictions is None:
            print("Initializing model...")
            if not hasattr(self, 'model') or self.model is None:
                if self.trainer is None:
                    self.trainer = CodeBERTTrainer(batch_size=8, epochs=3)
                
                # Check if we have a pre-trained model to load
                if self.model_path and os.path.exists(self.model_path):
                    # Load pre-trained model
                    print(f"Loading model from {self.model_path}")
                    self.model = self.trainer.load_model(self.model_path)
                    
                    # CRITICAL FIX: Ensure model is in evaluation mode after loading
                    if self.model is not None:
                        self.model.eval()
                        if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                            self.trainer.model.eval()
                        print("Model loaded successfully and set to evaluation mode")
                    else:
                        print("ERROR: Model loading returned None!")
                        raise ValueError("Failed to load model from specified path")
                else:
                    # Train a new model only if no pre-trained model exists
                    print("Training a new model")
                    from sklearn.model_selection import train_test_split
                    train_data, test_data = train_test_split(original_df, test_size=0.2, random_state=42)
                    
                    # Set trainer data
                    self.trainer.set_data(train_data)
                    
                    # Prepare data loaders
                    data_loaders = self.trainer.prepare_data(train_data, test_data)
                    
                    # Train the model
                    self.model = self.trainer.train_model(data_loaders, freeze_bert=False)
                    print("Model trained successfully")
            else:
                print("Using pre-loaded model")
                # CRITICAL FIX: Ensure the pre-loaded model is in evaluation mode
                if hasattr(self, 'model') and self.model is not None:
                    self.model.eval()
                if hasattr(self, 'trainer') and hasattr(self.trainer, 'model') and self.trainer.model is not None:
                    self.trainer.model.eval()
        else:
            print("Using loaded predictions from txt file, skipping model initialization")
            # Still need trainer for adversarial predictions if model_path is provided
            if self.model_path and not hasattr(self, 'trainer'):
                print("Loading model for adversarial prediction generation...")
                self.trainer = CodeBERTTrainer()
                self.model = self.trainer.load_model(self.model_path)
                if self.model is not None:
                    self.model.eval()
                    if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                        self.trainer.model.eval()
            elif self.model_path and hasattr(self, 'trainer') and self.trainer is None:
                print("Loading model for adversarial prediction generation...")
                self.trainer = CodeBERTTrainer()
                self.model = self.trainer.load_model(self.model_path)
                if self.model is not None:
                    self.model.eval()
                    if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                        self.trainer.model.eval()
            elif not self.model_path:
                print("Warning: No model path provided and using loaded predictions.")
                print("Adversarial predictions cannot be generated without a model.")
                # Initialize a dummy trainer to prevent AttributeError
                self.trainer = None
        
        # Calculate initial fitness scores
        print("Calculating initial fitness scores...")
        
        # Make sure the model is available for calculate_fitness
        # FIXED: Handle case when using loaded predictions and no trainer is available
        if hasattr(self, 'model') and self.model is not None:
            model = self.model
        elif hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'model'):
            model = self.trainer.model
        else:
            model = None
        
        # DEBUG: Validate model predictions on original data (only if trainer is available)
        if self.original_predictions is None and hasattr(self, 'trainer') and self.trainer is not None:
            print("\n===== MODEL PREDICTION VALIDATION =====")
            # Get counts of vulnerable samples in original data
            vulnerable_count = (original_df['label'] == 1).sum()
            print(f"Dataset has {vulnerable_count} labeled vulnerable samples out of {len(original_df)} total")
            
            # Check original predictions
            if hasattr(self.trainer, 'predict'):
                correct_predictions = 0
                vulnerable_correctly_identified = 0
                vulnerable_samples = original_df['label'] == 1
                
                for idx, row in original_df.iterrows():
                    code = row['functionSource']
                    true_label = row['label']
                    try:
                        pred = self.trainer.predict(code)
                        if isinstance(pred, dict) and 'prediction' in pred:
                            prediction = pred['prediction']
                            if prediction == true_label:
                                correct_predictions += 1
                                if true_label == 1:
                                    vulnerable_correctly_identified += 1
                            
                            # Print info for all vulnerable samples
                            if true_label == 1:
                                print(f"Vulnerable sample {idx}: Predicted as {'vulnerable' if prediction == 1 else 'benign'}")
                    except Exception as e:
                        print(f"Error predicting sample {idx}: {str(e)}")
                
                accuracy = correct_predictions / len(original_df) if len(original_df) > 0 else 0
                vulnerability_recall = vulnerable_correctly_identified / vulnerable_count if vulnerable_count > 0 else 0
                
                print(f"Model accuracy: {accuracy:.4f} ({correct_predictions}/{len(original_df)})")
                print(f"Vulnerability detection rate: {vulnerability_recall:.4f} ({vulnerable_correctly_identified}/{vulnerable_count})")
                
                if vulnerability_recall < 0.1:
                    print("WARNING: Model is detecting very few vulnerabilities, adversarial attacks will likely fail!")
                    print("Consider retraining the model or providing clearer vulnerable examples.")
                    
                    # Create more obvious vulnerable examples for testing
                    if vulnerable_correctly_identified == 0:
                        print("CRITICAL: No vulnerabilities detected. Creating synthetic examples for testing.")
                        # Create an obvious example with a known vulnerability
                        test_code = """void vulnerable_func() {
                            char buffer[10];
                            gets(buffer);  // Known buffer overflow
                            printf("%s", buffer);
                        }"""
                        
                        try:
                            pred = self.trainer.predict(test_code)
                            print(f"Test vulnerability prediction: {pred}")
                            if isinstance(pred, dict) and pred.get('prediction') != 1:
                                print("SEVERE WARNING: Model fails to detect even obvious vulnerabilities!")
                        except Exception as e:
                            print(f"Error in test prediction: {str(e)}")
        else:
            print("\n===== USING LOADED PREDICTIONS =====")
            print(f"Loaded {len(self.original_predictions)} predictions from txt file")
            print("Skipping model validation since predictions are pre-computed")
        
        # Track the actual attack success rates for the best code
        attack_success_rates = {}
        
        # Calculate fitness for each member of the population
        for adv_code in tqdm(list(self.population.keys()), desc="Initial fitness"):
            fitness, attack_rate = self.calculate_fitness(original_df, adv_code, model, return_attack_rate=True)
            self.population[adv_code] = fitness
            attack_success_rates[adv_code] = attack_rate
        
        # Diagnose the best initial adversarial code (only if trainer is available)
        if self.population and hasattr(self, 'trainer') and self.trainer is not None:
            best_initial_code = max(self.population.items(), key=lambda x: x[1])[0]
            print(f"\n=== DIAGNOSING BEST INITIAL ADVERSARIAL CODE ===")
            self.diagnose_attack_effectiveness(original_df, best_initial_code, model)
        
        # Free GPU memory
        free_gpu_memory()
        
        # Run generations
        best_fitness = max(self.population.values()) if self.population else 0
        best_code = max(self.population.items(), key=lambda x: x[1])[0] if self.population else None
        best_attack_rate = attack_success_rates.get(best_code, 0.0)
        
        for gen in range(self.max_generations):
            if self.verbose:
                print(f"\n=== Generation {gen+1}/{self.max_generations} ===")
                print(f"Best fitness so far: {best_fitness:.4f}")
                print(f"Best attack success rate: {best_attack_rate:.4f}")
            
            # Perform fuzzy clustering
            membership_weights, centroid_change = self.perform_fuzzy_clustering()
            
            # Select top clusters
            top_clusters = self.select_clusters()
            
            # Perform crossover
            offspring = self.perform_crossover(membership_weights, top_clusters)
            
            # Calculate fitness for offspring
            offspring_fitness = []
            for adv_code in tqdm(offspring, desc="Offspring fitness"):
                fitness, attack_rate = self.calculate_fitness(original_df, adv_code, model, return_attack_rate=True)
                offspring_fitness.append(fitness)
                attack_success_rates[adv_code] = attack_rate
            
            # Update population with offspring
            self.population = update_global_pop(offspring, self.population, offspring_fitness)
            
            # Check for new best fitness
            current_best = max(self.population.values())
            if current_best > best_fitness:
                best_fitness = current_best
                best_code = max(self.population.items(), key=lambda x: x[1])[0]
                best_attack_rate = attack_success_rates.get(best_code, 0.0)
                
                if self.verbose:
                    print(f"New best fitness: {best_fitness:.4f}")
                    print(f"New best attack success rate: {best_attack_rate:.4f}")
                    print(f"Best code snippet length: {len(best_code.splitlines())}")
            
            # Check for perfect attack (100% success rate)
            if best_fitness > 0.99 - self.penalty:  # Allow for length penalty
                if self.verbose:
                    print(f"Found optimal adversarial code with fitness {best_fitness:.4f}")
                break
            
            # Check for convergence
            if centroid_change < 1e-6:
                if self.verbose:
                    print(f"Converged after {gen+1} generations with best fitness {best_fitness:.4f}")
                break
        
        # Calculate the direct attack success rate with the best code
        # This matches the logic in the user's code sample
        print("\n=== Direct Attack Success Rate Calculation ===")
        
        # Use loaded predictions if available, otherwise get predictions from model
        if self.original_predictions is not None:
            print("Using loaded predictions for direct attack calculation...")
            original_predictions = self.original_predictions.copy()
        else:
            # Get predictions on original data using model
            def get_predictions(df):
                predictions = []
                if hasattr(self, 'trainer') and self.trainer is not None:
                    for _, row in df.iterrows():
                        code = row['functionSource']
                        try:
                            pred = self.trainer.predict(code)
                            if isinstance(pred, dict) and 'prediction' in pred:
                                predictions.append(pred['prediction'])
                            else:
                                predictions.append(0)
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"Error in prediction: {str(e)}")
                            predictions.append(0)
                else:
                    # No trainer available, return all zeros
                    predictions = [0] * len(df)
                return np.array(predictions)
            
            print("Getting original predictions from model...")
            original_predictions = get_predictions(original_df)
        
        # Generate adversarial predictions only if trainer is available
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Create a copy for adversarial testing
            adv_df = original_df.copy()
            vulnerable_samples = adv_df['label'] == 1
            num_vulnerable = vulnerable_samples.sum()
            
            print(f"Found {num_vulnerable} vulnerable samples for adversarial testing")
            
            # Insert adversarial code into vulnerable samples
            for idx in adv_df.index[vulnerable_samples]:
                orig_code = adv_df.loc[idx, 'functionSource']
                code_lines = orig_code.split('\n')
                insert_pos = min(15, max(1, len(code_lines) - 1))  # Try to use position 15 like in user's code
                code_lines.insert(insert_pos, best_code)
                adv_df.loc[idx, 'functionSource'] = '\n'.join(code_lines)
            
            # Get adversarial predictions
            print("Getting adversarial predictions...")
            adversarial_predictions = []
            for _, row in adv_df.iterrows():
                code = row['functionSource']
                try:
                    pred = self.trainer.predict(code)
                    if isinstance(pred, dict) and 'prediction' in pred:
                        adversarial_predictions.append(pred['prediction'])
                    else:
                        adversarial_predictions.append(0)
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"Error in adversarial prediction: {str(e)}")
                    adversarial_predictions.append(0)
            
            adversarial_predictions = np.array(adversarial_predictions)
            
            # Calculate direct attack success rate (percent of predictions that changed)
            vul_indices = np.where(original_df['label'] == 1)[0]
            prediction_changes = sum(1 for i in vul_indices 
                                   if original_predictions[i] != adversarial_predictions[i])
            
            direct_attack_success_rate = prediction_changes / len(vul_indices) if len(vul_indices) > 0 else 0
            
            # Count how many 1→0 changes (matching user's code logic - vulnerable to benign)
            one_to_zero = sum(1 for i in vul_indices
                              if original_predictions[i] == 1 and adversarial_predictions[i] == 0)
            
            one_to_zero_rate = one_to_zero / len(vul_indices) if len(vul_indices) > 0 else 0
        else:
            print("No trainer available for direct attack calculation, using fitness-based estimates")
            direct_attack_success_rate = best_attack_rate  # Use the best attack rate from fitness calculation
            one_to_zero_rate = best_attack_rate
            one_to_zero = int(best_attack_rate * (original_df['label'] == 1).sum())
            vul_indices = np.where(original_df['label'] == 1)[0]
            adversarial_predictions = original_predictions.copy() if self.original_predictions is not None else np.zeros(len(original_df))
        
        print("\n=== Final Attack Success Results ===")
        print(f"Overall Attack Success Rate (any change): {best_attack_rate:.4f}")
        if hasattr(self, 'trainer') and self.trainer is not None:
            print(f"Direct Attack Success Rate (vulnerable samples only): {direct_attack_success_rate:.4f}")
            print(f"Vulnerable to Benign Changes (1→0): {one_to_zero_rate:.4f} ({one_to_zero}/{len(vul_indices)})")
        print(f"Fitness Score: {best_fitness:.4f}")
        
        best_snippet_length = len(best_code.splitlines())
        length_penalty = self.penalty * best_snippet_length
        print(f"Length Penalty: {length_penalty:.4f}")
        print(f"Code Snippet Length: {best_snippet_length}")
        
        # Generate and save adversarial predictions with the best code
        print("\n=== GENERATING ADVERSARIAL PREDICTIONS ===")
        
        # Apply the best adversarial code to create adversarial dataset
        final_adv_df = original_df.copy()
        vulnerable_samples = final_adv_df['label'] == 1
        
        # Insert best adversarial code into vulnerable samples
        for idx in final_adv_df.index[vulnerable_samples]:
            orig_code = final_adv_df.loc[idx, 'functionSource']
            code_lines = orig_code.split('\n')
            insert_pos = min(15, max(1, len(code_lines) - 1))
            code_lines.insert(insert_pos, best_code)
            final_adv_df.loc[idx, 'functionSource'] = '\n'.join(code_lines)
        
        # Generate adversarial predictions
        if hasattr(self, 'trainer') and self.trainer is not None:
            print("Generating adversarial predictions with best code...")
            final_adversarial_predictions = []
            
            for _, row in tqdm(final_adv_df.iterrows(), desc="Generating adversarial predictions", total=len(final_adv_df)):
                code = row['functionSource']
                try:
                    pred = self.trainer.predict(code)
                    if isinstance(pred, dict) and 'prediction' in pred:
                        final_adversarial_predictions.append(pred['prediction'])
                    else:
                        final_adversarial_predictions.append(0)
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"Error in final adversarial prediction: {str(e)}")
                    final_adversarial_predictions.append(0)
            
            final_adversarial_predictions = np.array(final_adversarial_predictions)
            
            # Extract dataset name from original_data_path for consistent naming
            dataset_name = "test"  # Default
            if original_data_path:
                # Extract CWE ID from path like 'cwe399_test.csv'
                import re
                cwe_match = re.search(r'cwe(\d+)', os.path.basename(original_data_path).lower())
                if cwe_match:
                    dataset_name = f"cwe{cwe_match.group(1)}"
            
            # Save adversarial predictions
            adv_predictions_path = self.save_adversarial_predictions(
                final_adversarial_predictions, 
                dataset_name
            )
            
            # Calculate final adversarial attack statistics
            if self.original_predictions is not None:
                original_preds = self.original_predictions
            else:
                # Use original predictions from earlier calculation
                original_preds = original_predictions
            
            # Calculate attack effectiveness on final adversarial predictions
            total_changes = np.sum(original_preds != final_adversarial_predictions)
            vuln_to_benign = np.sum((original_preds == 1) & (final_adversarial_predictions == 0))
            benign_to_vuln = np.sum((original_preds == 0) & (final_adversarial_predictions == 1))
            
            # Calculate original model performance metrics
            original_accuracy = accuracy_score(original_df['label'], original_preds)
            original_precision = precision_score(original_df['label'], original_preds, average='binary', zero_division=0)
            original_recall = recall_score(original_df['label'], original_preds, average='binary', zero_division=0)
            original_f1 = f1_score(original_df['label'], original_preds, average='binary', zero_division=0)
            
            print(f"\n=== FINAL ADVERSARIAL ATTACK RESULTS ===")
            print(f"Total prediction changes: {total_changes}/{len(original_preds)} ({total_changes/len(original_preds):.4f})")
            print(f"Vulnerable→Benign changes: {vuln_to_benign}")
            print(f"Benign→Vulnerable changes: {benign_to_vuln}")
            print(f"Adversarial predictions saved to: {adv_predictions_path}")
            print(f"\n=== ORIGINAL MODEL PERFORMANCE ===")
            print(f"Accuracy: {original_accuracy:.4f}")
            print(f"Precision: {original_precision:.4f}")
            print(f"Recall: {original_recall:.4f}")
            print(f"F1-Score: {original_f1:.4f}")
            
            # Create results in the desired format
            results = {
                'best_adversarial_code': best_code,
                'best_fitness': best_fitness,
                'attack_success_rate': best_attack_rate,
                'original_f1_score': round(original_f1, 4),
                'original_accuracy': round(original_accuracy, 4),
                'original_precision': round(original_precision, 4),
                'original_recall': round(original_recall, 4),
                'parameters': {
                    'pop_size': self.pop_size,
                    'clusters': self.clusters,
                    'max_generations': self.max_generations,
                    'decay_rate': self.decay_rate,
                    'alpha': self.alpha,
                    'penalty': self.penalty
                },
                'adversarial_predictions_file': adv_predictions_path,
                'vulnerable_to_benign_changes': int(vuln_to_benign)
            }
        else:
            print("No model available for generating adversarial predictions")
            adv_predictions_path = None
            
            # For cases without model, we can't calculate detailed metrics
            # Provide basic results structure
            results = {
                'best_adversarial_code': best_code,
                'best_fitness': best_fitness,
                'attack_success_rate': best_attack_rate,
                'original_f1_score': 0.0,
                'original_accuracy': 0.0,
                'original_precision': 0.0,
                'original_recall': 0.0,
                'parameters': {
                    'pop_size': self.pop_size,
                    'clusters': self.clusters,
                    'max_generations': self.max_generations,
                    'decay_rate': self.decay_rate,
                    'alpha': self.alpha,
                    'penalty': self.penalty
                },
                'adversarial_predictions_file': None,
                'vulnerable_to_benign_changes': 0
            }
        
        # Extract CWE ID from original_data_path to create filename suffix
        if original_data_path:
            # Extract CWE ID from path like '/kaggle/input/eatvul/cwe399_test.csv'
            import re
            cwe_match = re.search(r'cwe(\d+)', os.path.basename(original_data_path).lower())
            if cwe_match:
                cwe_id = cwe_match.group(1)
                results_filename = f'adversarial_results_cwe{cwe_id}.json'
            else:
                results_filename = 'adversarial_results.json'
        else:
            results_filename = 'adversarial_results.json'
        
        # Save best adversarial code
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_filename}")
        
        return best_code, best_fitness

    def diagnose_attack_effectiveness(self, original_df, adversarial_code, model=None):
        """
        Detailed diagnosis of why an adversarial attack might be failing
        """
        print(f"\n=== ATTACK EFFECTIVENESS DIAGNOSIS ===")
        print(f"Adversarial code being tested:")
        print("-" * 50)
        print(adversarial_code)
        print("-" * 50)
        
        # Get model predictions function
        def get_prediction_details(code):
            try:
                pred = self.trainer.predict(code)
                return pred
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                return {'prediction': 0, 'confidence': 0.5, 'probabilities': [0.5, 0.5]}
        
        # Test on a few vulnerable samples
        vulnerable_samples = original_df[original_df['label'] == 1].head(10)
        
        print(f"\nTesting on {len(vulnerable_samples)} vulnerable samples:")
        
        successful_attacks = 0
        for idx, row in vulnerable_samples.iterrows():
            function_code = row['functionSource']
            
            # Get original prediction
            orig_pred = get_prediction_details(function_code)
            
            # If original is not predicted as vulnerable, skip
            if orig_pred['prediction'] != 1:
                print(f"Sample {idx}: Originally not detected as vulnerable (pred={orig_pred['prediction']}, conf={orig_pred['confidence']:.3f})")
                continue
            
            # Create adversarial version
            code_lines = function_code.split('\n')
            # Insert adversarial code at position 2 (after function declaration typically)
            insert_pos = min(2, len(code_lines) - 1)
            code_lines.insert(insert_pos, adversarial_code)
            adv_code = '\n'.join(code_lines)
            
            # Get adversarial prediction
            adv_pred = get_prediction_details(adv_code)
            
            # Check if attack was successful
            attack_success = (orig_pred['prediction'] == 1 and adv_pred['prediction'] == 0)
            if attack_success:
                successful_attacks += 1
            
            print(f"Sample {idx}:")
            print(f"  Original: pred={orig_pred['prediction']}, conf={orig_pred['confidence']:.3f}")
            print(f"  Adversarial: pred={adv_pred['prediction']}, conf={adv_pred['confidence']:.3f}")
            print(f"  Attack success: {attack_success}")
            print(f"  Confidence change: {orig_pred['confidence']:.3f} -> {adv_pred['confidence']:.3f}")
            
            # Show a snippet of the adversarial code
            print(f"  Adversarial code snippet:")
            adv_lines = adv_code.split('\n')
            for i, line in enumerate(adv_lines[max(0, insert_pos-1):insert_pos+3]):
                marker = ">>> " if i == 1 else "    "
                print(f"    {marker}{line}")
            print()
        
        attack_rate = successful_attacks / len(vulnerable_samples) if len(vulnerable_samples) > 0 else 0
        print(f"Overall attack success rate: {attack_rate:.4f} ({successful_attacks}/{len(vulnerable_samples)})")
        
        # Additional diagnostics
        print(f"\n=== ADDITIONAL DIAGNOSTICS ===")
        
        # Test if the adversarial code itself is detected as vulnerable
        test_func = f"""void test_function() {{
    {adversarial_code}
    return;
}}"""
        
        test_pred = get_prediction_details(test_func)
        print(f"Adversarial code in isolation:")
        print(f"  Prediction: {test_pred['prediction']} (0=benign, 1=vulnerable)")
        print(f"  Confidence: {test_pred['confidence']:.3f}")
        
        if test_pred['prediction'] == 0:
            print("  -> Adversarial code itself is not detected as vulnerable")
            print("  -> This might explain low attack success rates")
        else:
            print("  -> Adversarial code is detected as vulnerable when isolated")
            print("  -> The problem might be in how it's inserted into existing code")
        
        return attack_rate

    def load_predictions_from_txt(self, prediction_file_path):
        """
        Load model predictions from exported txt file
        
        Args:
            prediction_file_path: Path to the txt file containing predictions
            
        Returns:
            numpy array of predictions
        """
        if not os.path.exists(prediction_file_path):
            raise FileNotFoundError(f"Prediction file not found: {prediction_file_path}")
        
        predictions = []
        
        if self.verbose:
            print(f"\n=== LOADING PREDICTIONS FROM TXT ===")
            print(f"Loading predictions from: {prediction_file_path}")
        
        with open(prediction_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            index, prediction = parts
                            predictions.append(int(prediction))
                        else:
                            # Try space separator if tab doesn't work
                            parts = line.split()
                            if len(parts) == 2:
                                index, prediction = parts
                                predictions.append(int(prediction))
                            else:
                                print(f"Warning: Skipping malformed line {line_num + 1}: {line}")
                    except ValueError as e:
                        print(f"Warning: Error parsing line {line_num + 1}: {line} - {str(e)}")
        
        predictions = np.array(predictions)
        
        if self.verbose:
            print(f"Loaded {len(predictions)} predictions")
            print(f"Prediction distribution: {np.bincount(predictions)}")
            print(f"  0 (not vulnerable): {np.sum(predictions == 0)}")
            print(f"  1 (vulnerable): {np.sum(predictions == 1)}")
        
        self.original_predictions = predictions
        self.prediction_file_path = prediction_file_path
        
        return predictions
    
    def save_adversarial_predictions(self, adversarial_predictions, dataset_name="test"):
        """
        Save adversarial predictions to txt file
        
        Args:
            adversarial_predictions: Array of adversarial predictions
            dataset_name: Name to include in filename
            
        Returns:
            Path to the saved file
        """
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create filename
        if "cwe" in dataset_name.lower():
            filename = f"prediction_adv_{dataset_name}_{timestamp}.txt"
        else:
            filename = f"prediction_adv_cwe_{timestamp}.txt"
        
        # Determine output directory - handle read-only input directories
        output_dir = os.getcwd()  # Default to current working directory
        
        if self.prediction_file_path:
            input_dir = os.path.dirname(self.prediction_file_path)
            
            # Test if the input directory is writable
            try:
                test_file = os.path.join(input_dir, '.test_write_permission')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                # If we get here, the directory is writable
                output_dir = input_dir
                if self.verbose:
                    print(f"Using input directory for output: {output_dir}")
            except (OSError, PermissionError):
                # Directory is read-only, use current working directory
                output_dir = os.getcwd()
                if self.verbose:
                    print(f"Input directory is read-only, using current directory: {output_dir}")
        
        output_path = os.path.join(output_dir, filename)
        
        # Write predictions to file
        with open(output_path, 'w') as f:
            for idx, pred in enumerate(adversarial_predictions):
                f.write(f"{idx}\t{pred}\n")
        
        if self.verbose:
            print(f"Adversarial predictions exported to: {output_path}")
            print(f"Total adversarial predictions exported: {len(adversarial_predictions)}")
        
        return output_path
