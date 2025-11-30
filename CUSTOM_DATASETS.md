# Using Custom Datasets - Quick Guide

This guide explains how to use your own datasets instead of the default `technician_feedback.csv`.

## 📋 Dataset Requirements

Your CSV file should have at least two columns:
1. **Text column**: Contains the text/comments to analyze
2. **Label column**: Contains sentiment labels (e.g., positive, negative, neutral)

### Example CSV Format

```csv
feedback_text,sentiment
"Great product, very satisfied",positive
"Terrible experience, disappointed",negative
"It's okay, nothing special",neutral
```

## 🚀 Quick Start

### Method 1: Simple CLI (Recommended)

```bash
# Train with your dataset
python run_training.py --dataset path/to/your/data.csv

# Specify custom column names
python run_training.py --dataset data.csv --text-col review --label-col rating

# Train and evaluate in one command
python run_training.py --dataset data.csv --evaluate
```

### Method 2: Direct Script Usage

```bash
# Train models
python scripts/train_models.py --data-path path/to/your/data.csv --text-col review --label-col sentiment

# Evaluate models
python scripts/evaluate_models.py --data-path path/to/your/data.csv --text-col review --label-col sentiment
```

### Method 3: Streamlit Web App

1. Start the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. In the sidebar:
   - Select "Use existing dataset" and choose from available datasets
   - OR select "Upload new dataset" and upload your CSV file

3. The app will automatically detect text and sentiment columns

## 📝 Command-Line Arguments

### run_training.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to your CSV file | `data/technician_feedback.csv` |
| `--text-col` | Name of text column | `feedback_text` |
| `--label-col` | Name of label column | `sentiment` |
| `--generate` | Generate sample dataset if missing | False |
| `--evaluate` | Run evaluation after training | False |

### scripts/train_models.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path` | Path to CSV file | `data/technician_feedback.csv` |
| `--text-col` | Name of text column | `feedback_text` |
| `--label-col` | Name of label column | `sentiment` |
| `--generate` | Generate sample dataset if missing | False |

### scripts/evaluate_models.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path` | Path to CSV file | `data/technician_feedback.csv` |
| `--text-col` | Name of text column | `feedback_text` |
| `--label-col` | Name of label column | `sentiment` |

## 🔧 Configuration File

You can also update `config/config.yaml`:

```yaml
data:
  file_path: "data/my_custom_data.csv"
  text_column: "review_text"
  label_column: "rating"
```

## 💡 Examples

### Example 1: Product Reviews

```bash
# Your CSV: product_reviews.csv with columns 'review' and 'rating'
python run_training.py --dataset data/product_reviews.csv --text-col review --label-col rating --evaluate
```

### Example 2: Customer Feedback

```bash
# Your CSV: customer_feedback.csv with columns 'comment' and 'sentiment'
python run_training.py --dataset data/customer_feedback.csv --text-col comment --label-col sentiment
```

### Example 3: Multiple Datasets

```bash
# Train on dataset 1
python run_training.py --dataset data/dataset1.csv --text-col text --label-col label

# Evaluate on dataset 2
python scripts/evaluate_models.py --data-path data/dataset2.csv --text-col text --label-col label
```

## 📊 Supported Label Formats

The system automatically handles various sentiment labels:
- **Binary**: positive/negative
- **Ternary**: positive/negative/neutral
- **Multi-class**: any custom labels (will be automatically encoded)

## ⚠️ Common Issues

### Issue: Column not found

**Error**: `ValueError: Text column 'xyz' not found in dataset`

**Solution**: Check your column names and use `--text-col` to specify the correct name

### Issue: Dataset not found

**Error**: `Error: Dataset not found at path/to/file.csv`

**Solution**: 
- Check the file path is correct
- Use `--generate` flag to create a sample dataset first

### Issue: Labels not recognized

**Solution**: Ensure your label column contains consistent values (e.g., don't mix "positive", "Positive", "pos")

## 🎯 Best Practices

1. **Balanced Dataset**: Try to have similar numbers of samples for each sentiment
2. **Clean Data**: Remove duplicates and empty rows before training
3. **Consistent Labels**: Use consistent sentiment labels throughout your dataset
4. **Encoding**: Save your CSV in UTF-8 encoding to avoid character issues
5. **Size**: Use at least 100 samples per sentiment class for good results

## 📞 Need Help?

- Check the main README.md for project documentation
- View example datasets in the `data/` folder
- See the configuration in `config/config.yaml`

## 🔄 Workflow

```
1. Prepare your CSV file
   ↓
2. Train models with your dataset
   python run_training.py --dataset your_data.csv --evaluate
   ↓
3. View results in models/ and outputs/ folders
   ↓
4. Use trained models in Streamlit app
   streamlit run app/streamlit_app.py
```

---

Happy analyzing! 🎉

