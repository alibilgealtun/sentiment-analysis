"""
Evaluation Module

This module provides functions for evaluating sentiment classification models,
including metrics calculation, visualization, and model comparison.

Author: Sentiment Analysis Team
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize


# Set style for all visualizations
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#3498db',
    'primary': '#2c3e50',
    'secondary': '#95a5a6'
}


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy for multi-class metrics.
    
    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and F1-score.
    
    Examples
    --------
    >>> y_true = ['positive', 'negative', 'neutral']
    >>> y_pred = ['positive', 'negative', 'positive']
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(metrics['accuracy'])
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    Generate a classification report.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of label names.
    output_dict : bool, default=False
        If True, return report as dictionary.
    
    Returns
    -------
    str or dict
        Classification report.
    
    Examples
    --------
    >>> report = get_classification_report(y_true, y_pred)
    >>> print(report)
    """
    return classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=output_dict,
        zero_division=0
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None,
    normalize: bool = False
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of label names for axes.
    title : str, default='Confusion Matrix'
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.
    cmap : str, default='Blues'
        Colormap for heatmap.
    save_path : str, optional
        Path to save the figure.
    normalize : bool, default=False
        Whether to normalize the confusion matrix.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Examples
    --------
    >>> fig = plot_confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels if labels else 'auto',
        yticklabels=labels if labels else 'auto',
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    title: str = 'ROC Curves',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Prediction probabilities (shape: n_samples x n_classes).
    classes : list
        List of class names.
    title : str, default='ROC Curves'
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Examples
    --------
    >>> fig = plot_roc_curves(y_true, y_prob, classes=['positive', 'negative', 'neutral'])
    >>> plt.show()
    """
    # Binarize labels
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_name in enumerate(classes):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, class_name in enumerate(classes):
        ax.plot(
            fpr[class_name],
            tpr[class_name],
            color=colors[i],
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[class_name]:.3f})'
        )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    title: str = 'Precision-Recall Curves',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class classification.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Prediction probabilities.
    classes : list
        List of class names.
    title : str, default='Precision-Recall Curves'
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Binarize labels
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    # Compute PR curve for each class
    precision = {}
    recall = {}
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, class_name in enumerate(classes):
        precision[class_name], recall[class_name], _ = precision_recall_curve(
            y_bin[:, i], y_prob[:, i]
        )
        ax.plot(
            recall[class_name],
            precision[class_name],
            color=colors[i],
            lw=2,
            label=f'{class_name}'
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")
    
    return fig


def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart comparing multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and metric dictionaries as values.
        Example: {'Naive Bayes': {'accuracy': 0.85, 'f1_score': 0.82}, ...}
    metrics : list, default=['accuracy', 'precision', 'recall', 'f1_score']
        Metrics to compare.
    title : str, default='Model Comparison'
        Plot title.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    
    Examples
    --------
    >>> results = {
    ...     'Naive Bayes': {'accuracy': 0.85, 'f1_score': 0.82},
    ...     'SVM': {'accuracy': 0.88, 'f1_score': 0.86}
    ... }
    >>> fig = compare_models(results)
    >>> plt.show()
    """
    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Prepare data
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    color: str = '#3498db',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    feature_importance : dict
        Dictionary mapping feature names to importance scores.
    top_n : int, default=20
        Number of top features to display.
    title : str, default='Feature Importance'
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size.
    color : str, default='#3498db'
        Bar color.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Sort and get top N features
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features = [f[0] for f in sorted_features]
    importance = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_class_distribution(
    y: np.ndarray,
    title: str = 'Class Distribution',
    figsize: Tuple[int, int] = (10, 5),
    plot_type: str = 'both',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution as bar chart and/or pie chart.
    
    Parameters
    ----------
    y : array-like
        Labels array.
    title : str, default='Class Distribution'
        Plot title.
    figsize : tuple, default=(10, 5)
        Figure size.
    plot_type : str, default='both'
        Type of plot: 'bar', 'pie', or 'both'.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Count classes
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Define colors for sentiments
    colors = [COLORS.get(label, '#3498db') for label in unique]
    
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart
        ax1.bar(unique, counts, color=colors, alpha=0.8)
        ax1.set_xlabel('Sentiment', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Bar Chart', fontsize=12)
        for i, (label, count) in enumerate(zip(unique, counts)):
            ax1.text(i, count + max(counts) * 0.02, str(count), ha='center', fontsize=10)
        
        # Pie chart
        ax2.pie(
            counts,
            labels=unique,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(unique)
        )
        ax2.set_title('Pie Chart', fontsize=12)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    elif plot_type == 'bar':
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(unique, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        for i, (label, count) in enumerate(zip(unique, counts)):
            ax.text(i, count + max(counts) * 0.02, str(count), ha='center', fontsize=10)
    
    else:  # pie
        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            counts,
            labels=unique,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(unique)
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    return fig


def plot_training_history(
    history,
    metrics: List[str] = ['loss', 'accuracy'],
    title: str = 'Training History',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history from Keras model.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history object.
    metrics : list, default=['loss', 'accuracy']
        Metrics to plot.
    title : str, default='Training History'
        Plot title.
    figsize : tuple, default=(12, 5)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history.history:
            ax.plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.set_title(f'{metric.title()} over Epochs', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score']
) -> pd.DataFrame:
    """
    Create a formatted results table.
    
    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and metric dictionaries as values.
    metrics : list, default=['accuracy', 'precision', 'recall', 'f1_score']
        Metrics to include in the table.
    
    Returns
    -------
    pd.DataFrame
        Formatted results table.
    
    Examples
    --------
    >>> results = {
    ...     'Naive Bayes': {'accuracy': 0.85, 'f1_score': 0.82},
    ...     'SVM': {'accuracy': 0.88, 'f1_score': 0.86}
    ... }
    >>> df = create_results_table(results)
    >>> print(df)
    """
    data = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        for metric in metrics:
            value = model_results.get(metric, 0)
            row[metric.replace('_', ' ').title()] = f'{value:.4f}'
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def get_best_model(
    results: Dict[str, Dict[str, float]],
    metric: str = 'f1_score'
) -> Tuple[str, float]:
    """
    Get the best performing model based on a metric.
    
    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and metric dictionaries as values.
    metric : str, default='f1_score'
        Metric to use for comparison.
    
    Returns
    -------
    tuple
        (model_name, score) of the best model.
    """
    best_model = None
    best_score = -1
    
    for model_name, model_results in results.items():
        score = model_results.get(metric, 0)
        if score > best_score:
            best_score = score
            best_model = model_name
    
    return best_model, best_score


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Sample data
    y_true = np.array(['positive', 'negative', 'neutral', 'positive', 'negative'] * 20)
    y_pred = np.array(['positive', 'negative', 'positive', 'positive', 'negative'] * 20)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("=== Metrics ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(get_classification_report(y_true, y_pred))
    
    # Model comparison
    results = {
        'Naive Bayes': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1_score': 0.82},
        'SVM': {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.88, 'f1_score': 0.86},
        'Logistic Regression': {'accuracy': 0.86, 'precision': 0.85, 'recall': 0.86, 'f1_score': 0.85},
        'Random Forest': {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.87, 'f1_score': 0.85}
    }
    
    print("\n=== Results Table ===")
    print(create_results_table(results))
    
    print("\n=== Best Model ===")
    best_model, best_score = get_best_model(results)
    print(f"{best_model}: F1-Score = {best_score:.4f}")
