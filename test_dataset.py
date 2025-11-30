#!/usr/bin/env python
"""
Test Custom Dataset Loading

This script tests the custom dataset functionality without requiring 
all dependencies to be installed.

Author: Sentiment Analysis Team
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_load_data(filepath, text_column='feedback_text', label_column='sentiment'):
    """Test loading data from a CSV file."""
    print(f"\n{'=' * 60}")
    print(f"Testing Data Loading")
    print(f"{'=' * 60}\n")
    
    print(f"📁 File path: {filepath}")
    print(f"📝 Text column: {text_column}")
    print(f"🏷️  Label column: {label_column}\n")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return False
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded CSV file")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}\n")
        
        if text_column not in df.columns:
            print(f"❌ Error: Text column '{text_column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        if label_column not in df.columns:
            print(f"❌ Error: Label column '{label_column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        texts = df[text_column].values
        labels = df[label_column].values
        
        print(f"✅ Found text column: '{text_column}'")
        print(f"   Sample: {texts[0][:50]}...\n")
        
        print(f"✅ Found label column: '{label_column}'")
        print(f"   Unique labels: {list(df[label_column].unique())}")
        print(f"   Label distribution:")
        print(df[label_column].value_counts().to_string())
        print()
        
        print(f"✅ All checks passed!")
        print(f"   Total samples: {len(texts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test custom dataset loading')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/technician_feedback.csv',
        help='Path to the CSV file'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        default='feedback_text',
        help='Name of the text column'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='sentiment',
        help='Name of the label column'
    )
    
    args = parser.parse_args()
    
    success = test_load_data(args.data_path, args.text_col, args.label_col)
    
    if success:
        print(f"\n{'=' * 60}")
        print("🎉 Test Completed Successfully!")
        print(f"{'=' * 60}\n")
        print("💡 Next steps:")
        print("   - Train models: python scripts/train_models.py --data-path", args.data_path)
        print("   - Or use CLI: python run_training.py --dataset", args.data_path)
        print()
    else:
        print(f"\n{'=' * 60}")
        print("❌ Test Failed")
        print(f"{'=' * 60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

