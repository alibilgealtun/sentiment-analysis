#!/usr/bin/env python
"""
Sentiment Analysis Training CLI

A simplified command-line interface for training sentiment analysis models
with custom datasets.

Usage:
    # Train with default dataset
    python run_training.py

    # Train with custom dataset
    python run_training.py --dataset path/to/your/data.csv

    # Train with custom dataset and column names
    python run_training.py --dataset data.csv --text-col review --label-col rating

Author: Sentiment Analysis Team
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Train sentiment analysis models with your dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default dataset (data/technician_feedback.csv)
  python run_training.py
  
  # Train with custom dataset
  python run_training.py --dataset data/my_reviews.csv
  
  # Specify custom column names
  python run_training.py --dataset data.csv --text-col review_text --label-col sentiment_label
  
  # Generate default dataset if missing
  python run_training.py --generate
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/technician_feedback.csv',
        help='Path to your CSV dataset (default: data/technician_feedback.csv)'
    )

    parser.add_argument(
        '--train-dataset',
        type=str,
        help='Path to the CSV used for training (overrides --dataset)'
    )

    parser.add_argument(
        '--test-dataset',
        type=str,
        help='Path to the CSV used for testing/evaluation'
    )

    parser.add_argument(
        '--text-col',
        type=str,
        default='feedback_text',
        help='Name of the text column in your CSV (default: feedback_text)'
    )

    parser.add_argument(
        '--label-col',
        type=str,
        default='sentiment',
        help='Name of the label/sentiment column (default: sentiment)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        help='Custom name for the trained models (e.g., "Technician_Feedback", "Turkish_Sentiment")'
    )

    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate default dataset if file not found'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation after training'
    )

    args = parser.parse_args()

    train_dataset = args.train_dataset or args.dataset
    test_dataset = args.test_dataset

    if not os.path.exists(train_dataset) and not args.generate:
        print(f"❌ Error: Training dataset not found at {train_dataset}")
        print("💡 Tip: Use --generate flag to create a sample dataset, or provide a valid path")
        sys.exit(1)

    if test_dataset and not os.path.exists(test_dataset):
        print(f"❌ Error: Test dataset not found at {test_dataset}")
        sys.exit(1)

    print("=" * 60)
    print("🤖 SENTIMENT ANALYSIS MODEL TRAINING")
    print("=" * 60)
    print(f"\n📁 Training dataset: {train_dataset}")
    if test_dataset:
        print(f"📁 Test dataset: {test_dataset}")
    print(f"📝 Text column: {args.text_col}")
    print(f"🏷️  Label column: {args.label_col}")
    print()

    # Auto-generate model name if not provided
    model_name = args.model_name
    if not model_name:
        # Extract dataset name from path (remove .csv extension)
        dataset_basename = os.path.basename(train_dataset)
        model_name = os.path.splitext(dataset_basename)[0]

    print(f"🏷️  Model name: {model_name}")

    # Build training command
    train_cmd = [
        sys.executable,
        'scripts/train_models.py',
        '--train-data-path', train_dataset,
        '--text-col', args.text_col,
        '--label-col', args.label_col,
        '--model-name', model_name
    ]

    if test_dataset:
        train_cmd.extend(['--test-data-path', test_dataset])

    if args.generate:
        train_cmd.append('--generate')

    # Run training
    print("🚀 Starting model training...")
    print("-" * 60)
    result = subprocess.run(train_cmd)

    if result.returncode != 0:
        print("\n❌ Training failed!")
        sys.exit(1)

    print("\n✅ Training completed successfully!")

    # Run evaluation if requested
    if args.evaluate:
        print("\n" + "=" * 60)
        print("📊 RUNNING MODEL EVALUATION")
        print("=" * 60 + "\n")

        eval_cmd = [
            sys.executable,
            'scripts/evaluate_models.py',
            '--data-path', train_dataset,
            '--text-col', args.text_col,
            '--label-col', args.label_col
        ]

        if test_dataset:
            eval_cmd.extend(['--test-data-path', test_dataset])

        result = subprocess.run(eval_cmd)

        if result.returncode != 0:
            print("\n❌ Evaluation failed!")
            sys.exit(1)

        print("\n✅ Evaluation completed successfully!")

    print("\n" + "=" * 60)
    print("🎉 ALL TASKS COMPLETED!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   - View results in models/model_results.csv")
    if test_dataset:
        print("   - Metrics reflect your provided test dataset")
    if args.evaluate:
        print("   - Check visualizations in outputs/ folder")
    print("   - Run the web app: streamlit run app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
