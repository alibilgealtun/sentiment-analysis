"""
Sentiment Analysis Package for Technician Feedback Classification

This package provides tools for:
- Generating synthetic technician feedback data
- Text preprocessing and feature extraction
- Training and evaluating sentiment classification models
- Visualizing results and model performance

Author: Sentiment Analysis Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sentiment Analysis Team"

# Lazy imports to avoid loading all dependencies at package import
__all__ = [
    "generate_technician_feedback",
    "TextPreprocessor",
    "TFIDFExtractor",
    "CountVectorizerExtractor",
    "NaiveBayesClassifier",
    "SVMClassifier",
    "LogisticRegressionClassifier",
    "RandomForestClassifier",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "compare_models",
    "load_data",
    "save_model",
    "load_model"
]


def __getattr__(name):
    """Lazy import of submodules and their components."""
    if name == "generate_technician_feedback":
        from .data_generator import generate_technician_feedback
        return generate_technician_feedback
    elif name == "TextPreprocessor":
        from .preprocessing import TextPreprocessor
        return TextPreprocessor
    elif name == "TFIDFExtractor":
        from .feature_extraction import TFIDFExtractor
        return TFIDFExtractor
    elif name == "CountVectorizerExtractor":
        from .feature_extraction import CountVectorizerExtractor
        return CountVectorizerExtractor
    elif name == "NaiveBayesClassifier":
        from .models import NaiveBayesClassifier
        return NaiveBayesClassifier
    elif name == "SVMClassifier":
        from .models import SVMClassifier
        return SVMClassifier
    elif name == "LogisticRegressionClassifier":
        from .models import LogisticRegressionClassifier
        return LogisticRegressionClassifier
    elif name == "RandomForestClassifier":
        from .models import RandomForestClassifier
        return RandomForestClassifier
    elif name == "calculate_metrics":
        from .evaluation import calculate_metrics
        return calculate_metrics
    elif name == "plot_confusion_matrix":
        from .evaluation import plot_confusion_matrix
        return plot_confusion_matrix
    elif name == "plot_roc_curves":
        from .evaluation import plot_roc_curves
        return plot_roc_curves
    elif name == "compare_models":
        from .evaluation import compare_models
        return compare_models
    elif name == "load_data":
        from .utils import load_data
        return load_data
    elif name == "save_model":
        from .utils import save_model
        return save_model
    elif name == "load_model":
        from .utils import load_model
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
