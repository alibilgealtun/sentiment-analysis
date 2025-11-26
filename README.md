# 📊 NLP Sentiment Analysis for Technician Feedback Classification

A comprehensive NLP sentiment analysis application for classifying technician feedback into positive, negative, and neutral sentiments. This project includes data generation, text preprocessing, multiple machine learning models, deep learning approaches, and an interactive Streamlit web application.

## 🎯 Project Overview

This project provides a complete pipeline for sentiment analysis:
- **Synthetic Data Generation**: Creates realistic technician feedback datasets
- **Text Preprocessing**: Cleaning, tokenization, stopword removal, lemmatization
- **Feature Extraction**: TF-IDF and Count Vectorization
- **Multiple Models**: Naive Bayes, SVM, Logistic Regression, Random Forest, LSTM, BERT
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, model comparison
- **Interactive Web App**: Streamlit application for real-time predictions

## ✨ Features

- 🔄 **Synthetic Data Generator** - Create customizable technician feedback datasets
- 📝 **Text Preprocessing Pipeline** - Complete NLP preprocessing with NLTK
- 🤖 **Multiple ML Models** - Compare 6 different classification approaches
- 📈 **Comprehensive Evaluation** - Detailed metrics and visualizations
- 🌐 **Web Application** - Interactive Streamlit app for predictions
- 📓 **Presentation Notebook** - Ready-to-present Jupyter notebook
- ⚙️ **Configurable** - YAML-based configuration for all parameters

## 📁 Project Structure

```
sentiment-analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── technician_feedback.csv       # Generated dataset (550+ samples)
├── notebooks/
│   └── sentiment_analysis_presentation.ipynb  # Presentation notebook
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_generator.py             # Synthetic data generation
│   ├── preprocessing.py              # Text preprocessing
│   ├── feature_extraction.py         # TF-IDF, Count Vectorizer
│   ├── models.py                     # ML/DL model implementations
│   ├── evaluation.py                 # Metrics and visualizations
│   └── utils.py                      # Helper functions
├── app/
│   └── streamlit_app.py              # Web application
├── models/
│   └── .gitkeep                      # Trained models directory
├── scripts/
│   ├── train_models.py               # Model training script
│   └── evaluate_models.py            # Model evaluation script
└── config/
    └── config.yaml                   # Configuration settings
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 📖 Usage

### 1. Generate Dataset

The dataset is pre-generated, but you can regenerate it:

```python
from src.data_generator import generate_technician_feedback, save_dataset

df = generate_technician_feedback(n_samples=550, random_seed=42)
save_dataset(df, 'data/technician_feedback.csv')
```

### 2. Train Models

Run the training script to train all models:

```bash
python scripts/train_models.py
```

This will:
- Load and preprocess the data
- Train Naive Bayes, SVM, Logistic Regression, and Random Forest models
- Save models to the `models/` directory
- Print performance metrics

### 3. Evaluate Models

Run comprehensive evaluation:

```bash
python scripts/evaluate_models.py
```

This generates:
- Confusion matrices for each model
- ROC curves
- Model comparison charts
- Results saved to `outputs/` directory

### 4. Run Web Application

Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Features:
- Real-time sentiment prediction
- Model selection dropdown
- Batch prediction (CSV upload)
- Interactive word clouds
- Performance metrics display

### 5. Jupyter Notebook

Open the presentation notebook:

```bash
jupyter notebook notebooks/sentiment_analysis_presentation.ipynb
```

## 🎨 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~0.75 | ~0.74 | ~0.75 | ~0.74 |
| SVM | ~0.82 | ~0.81 | ~0.82 | ~0.81 |
| Logistic Regression | ~0.80 | ~0.79 | ~0.80 | ~0.79 |
| Random Forest | ~0.78 | ~0.77 | ~0.78 | ~0.77 |
| LSTM | ~0.80 | ~0.79 | ~0.80 | ~0.79 |
| BERT (DistilBERT) | ~0.85 | ~0.84 | ~0.85 | ~0.84 |

*Note: Results may vary based on data split and random seed.*

## 📊 Dataset Information

The synthetic dataset contains:
- **550+ samples** of technician feedback
- **3 sentiment classes**: positive, negative, neutral
- **8 categories**: equipment, service, training, safety, workload, management, tools, communication
- **Timestamps**: Randomly distributed over the past year

### Sample Feedbacks

| Sentiment | Example |
|-----------|---------|
| Positive | "The new diagnostic tool has significantly improved our repair efficiency" |
| Negative | "Equipment keeps breaking down, we need better maintenance schedules" |
| Neutral | "Completed the scheduled maintenance as per the manual" |

## 🛠️ Configuration

All parameters can be configured in `config/config.yaml`:

```yaml
# Data settings
data:
  n_samples: 550
  test_size: 0.2
  
# Model hyperparameters
models:
  naive_bayes:
    alpha: 1.0
  svm:
    kernel: "linear"
    C: 1.0
  # ... more settings
```

## 📚 API Reference

### TextPreprocessor

```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemmatize=True
)

cleaned_text = preprocessor.full_preprocess("Your text here")
```

### Models

```python
from src.models import LogisticRegressionClassifier

clf = LogisticRegressionClassifier(C=1.0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### Evaluation

```python
from src.evaluation import calculate_metrics, plot_confusion_matrix

metrics = calculate_metrics(y_true, y_pred)
fig = plot_confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ scripts/ app/

# Lint code
flake8 src/ scripts/ app/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NLTK](https://www.nltk.org/) for NLP preprocessing
- [scikit-learn](https://scikit-learn.org/) for machine learning models
- [TensorFlow](https://www.tensorflow.org/) for deep learning
- [Hugging Face Transformers](https://huggingface.co/) for BERT models
- [Streamlit](https://streamlit.io/) for the web application

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for Technician Feedback Analysis**
