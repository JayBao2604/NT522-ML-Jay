#!/usr/bin/env python3
"""
Check training data class distribution
"""

import pandas as pd
import os

def check_data_distribution():
    """Check class distribution in training data"""
    
    print("🔍 CHECKING TRAINING DATA DISTRIBUTION")
    print("="*50)
    
    # Common paths where training data might be
    possible_paths = [
        "../data/train.csv",
        "../data/training.csv", 
        "data/train.csv",
        "train.csv",
        "../train.csv"
    ]
    
    data_found = False
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            
            try:
                df = pd.read_csv(path)
                print(f"Dataset shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                # Check for label column
                label_cols = ['label', 'labels', 'target', 'y', 'vulnerable']
                label_col = None
                
                for col in label_cols:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    print(f"\nLabel column: '{label_col}'")
                    
                    # Class distribution
                    value_counts = df[label_col].value_counts()
                    print(f"\nClass distribution:")
                    for label, count in value_counts.items():
                        percentage = (count / len(df)) * 100
                        print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
                    
                    # Check for severe imbalance
                    if len(value_counts) == 2:
                        minority_count = min(value_counts.values)
                        majority_count = max(value_counts.values)
                        imbalance_ratio = majority_count / minority_count
                        
                        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
                        
                        if imbalance_ratio > 10:
                            print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
                            print("   This explains why your model only predicts the majority class")
                        elif imbalance_ratio > 3:
                            print("⚠️  Moderate class imbalance detected")
                        else:
                            print("✅ Reasonable class balance")
                
                else:
                    print("❌ No label column found")
                    print("Available columns:", list(df.columns))
                
                data_found = True
                break
                
            except Exception as e:
                print(f"❌ Error reading {path}: {str(e)}")
    
    if not data_found:
        print("❌ No training data found in common locations")
        print("Please check where your training data is located")
        
        # List current directory contents
        print(f"\nCurrent directory contents:")
        try:
            files = os.listdir(".")
            for f in files:
                if f.endswith(('.csv', '.txt', '.json')):
                    print(f"  {f}")
        except:
            pass

if __name__ == "__main__":
    check_data_distribution() 