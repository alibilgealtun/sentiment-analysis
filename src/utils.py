"""
Utilities Module

This module provides helper functions for data loading, model saving/loading,
and other utility operations.

Author: Sentiment Analysis Team
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import joblib


def load_data(
    filepath: str,
    text_column: str = 'feedback_text',
    label_column: str = 'sentiment',
    encoding: str = 'utf-8'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    text_column : str, default='feedback_text'
        Name of the text column.
    label_column : str, default='sentiment'
        Name of the label column.
    encoding : str, default='utf-8'
        File encoding.
    
    Returns
    -------
    tuple
        (DataFrame, texts array, labels array)
    
    Examples
    --------
    >>> df, texts, labels = load_data('data/technician_feedback.csv')
    >>> print(f"Loaded {len(texts)} samples")
    """
    df = pd.read_csv(filepath, encoding=encoding)
    
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    texts = df[text_column].values
    labels = df[label_column].values
    
    print(f"Loaded {len(texts)} samples from {filepath}")
    print(f"Label distribution:\n{df[label_column].value_counts()}")
    
    return df, texts, labels


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model to a file using joblib.
    
    Parameters
    ----------
    model : object
        Model object to save.
    filepath : str
        Path to save the model.
    
    Examples
    --------
    >>> save_model(trained_model, 'models/my_model.joblib')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a model from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the model file.
    
    Returns
    -------
    object
        Loaded model.
    
    Examples
    --------
    >>> model = load_model('models/my_model.joblib')
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def create_word_cloud(
    texts: List[str],
    sentiment: Optional[str] = None,
    max_words: int = 100,
    figsize: Tuple[int, int] = (12, 8),
    background_color: str = 'white',
    colormap: str = 'viridis',
    save_path: Optional[str] = None
):
    """
    Create a word cloud from texts.
    
    Parameters
    ----------
    texts : list
        List of texts.
    sentiment : str, optional
        Sentiment label for title.
    max_words : int, default=100
        Maximum number of words.
    figsize : tuple, default=(12, 8)
        Figure size.
    background_color : str, default='white'
        Background color.
    colormap : str, default='viridis'
        Colormap for words.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("wordcloud and matplotlib are required for word cloud generation")
    
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        random_state=42
    ).generate(combined_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    if sentiment:
        ax.set_title(f'Word Cloud - {sentiment.title()} Sentiment', fontsize=16, fontweight='bold')
    else:
        ax.set_title('Word Cloud', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
    return fig


def get_sample_predictions(
    model,
    vectorizer,
    texts: List[str],
    n_samples: int = 5
) -> pd.DataFrame:
    """
    Get sample predictions from a model.
    
    Parameters
    ----------
    model : object
        Trained model with predict and predict_proba methods.
    vectorizer : object
        Fitted vectorizer for text transformation.
    texts : list
        List of texts to predict.
    n_samples : int, default=5
        Number of samples to show.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with texts, predictions, and confidence scores.
    """
    # Select random samples
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    sample_texts = [texts[i] for i in indices]
    
    # Transform and predict
    X = vectorizer.transform(sample_texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results
    results = []
    for text, pred, probs in zip(sample_texts, predictions, probabilities):
        confidence = max(probs) * 100
        results.append({
            'Text': text[:100] + '...' if len(text) > 100 else text,
            'Prediction': pred,
            'Confidence': f'{confidence:.1f}%'
        })
    
    return pd.DataFrame(results)


def split_data_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, ...]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Parameters
    ----------
    X : array-like
        Features or texts.
    y : array-like
        Labels.
    test_size : float, default=0.2
        Proportion for test set.
    val_size : float, default=0.1
        Proportion for validation set (from training data).
    random_state : int, default=42
        Random seed.
    
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, default='config/config.yaml'
        Path to configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load config files")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"Random seeds set to {seed}")


def print_section_header(title: str, char: str = '=', width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Parameters
    ----------
    title : str
        Section title.
    char : str, default='='
        Character for decoration.
    width : int, default=60
        Total width of the header.
    """
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage.
    
    Parameters
    ----------
    value : float
        Decimal value (e.g., 0.85).
    decimals : int, default=2
        Number of decimal places.
    
    Returns
    -------
    str
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"


def get_model_summary(model) -> Dict:
    """
    Get a summary of a model's configuration.
    
    Parameters
    ----------
    model : object
        Model object.
    
    Returns
    -------
    dict
        Dictionary with model information.
    """
    summary = {
        'type': type(model).__name__,
        'is_fitted': getattr(model, 'is_fitted', 'Unknown'),
    }
    
    # Get model-specific parameters
    if hasattr(model, 'get_params'):
        summary['parameters'] = model.get_params()
    
    return summary


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str
        Directory path.
    """
    os.makedirs(path, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Parameters
    ----------
    filepath : str
        Path to file.
    
    Returns
    -------
    str
        Human-readable file size.
    """
    if not os.path.exists(filepath):
        return "File not found"
    
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.2f} TB"


if __name__ == "__main__":
    # Example usage
    print_section_header("Utility Functions Demo")
    
    # Test percentage formatting
    print(f"0.8523 as percentage: {format_percentage(0.8523)}")
    print(f"0.8523 as percentage (3 decimals): {format_percentage(0.8523, 3)}")
    
    # Test directory creation
    ensure_directory('/tmp/test_dir')
    print("Test directory created")
    
    # Set random seeds
    set_random_seeds(42)
