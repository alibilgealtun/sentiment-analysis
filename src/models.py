"""
Models Module

This module provides sentiment classification models for technician feedback.
It includes traditional ML models (Naive Bayes, SVM, Logistic Regression, Random Forest)
and deep learning models (LSTM, BERT).

Author: Sentiment Analysis Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder


class BaseSentimentClassifier:
    """
    Base class for sentiment classifiers.
    
    This class provides common functionality for all sentiment
    classification models.
    
    Attributes
    ----------
    model : object
        The underlying sklearn/keras model.
    is_fitted : bool
        Whether the model has been trained.
    label_encoder : LabelEncoder
        Encoder for sentiment labels.
    classes_ : array
        Unique class labels.
    """
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.classes_ = None
    
    def fit(self, X, y) -> 'BaseSentimentClassifier':
        """
        Train the model.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : array-like
            Features to predict.
        
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters
        ----------
        X : array-like
            Features to predict.
        
        Returns
        -------
        np.ndarray
            Prediction probabilities.
        """
        raise NotImplementedError("Subclasses must implement predict_proba()")
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from.
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes_']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")


class NaiveBayesClassifier(BaseSentimentClassifier):
    """
    Multinomial Naive Bayes Classifier for sentiment analysis.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
    
    Examples
    --------
    >>> clf = NaiveBayesClassifier()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)
    
    def fit(self, X, y) -> 'NaiveBayesClassifier':
        """Train the Naive Bayes model."""
        self.classes_ = np.unique(y)
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def tune_hyperparameters(
        self,
        X,
        y,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """
        Tune hyperparameters using GridSearchCV.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        param_grid : dict, optional
            Parameter grid for search.
        cv : int, default=5
            Number of cross-validation folds.
        
        Returns
        -------
        dict
            Best parameters found.
        """
        if param_grid is None:
            param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
        
        y_encoded = self.label_encoder.fit_transform(y)
        grid_search = GridSearchCV(
            MultinomialNB(),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_search.fit(X, y_encoded)
        
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.classes_ = np.unique(y)
        
        return grid_search.best_params_


class SVMClassifier(BaseSentimentClassifier):
    """
    Support Vector Machine Classifier for sentiment analysis.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type ('linear', 'rbf', 'poly').
    C : float, default=1.0
        Regularization parameter.
    probability : bool, default=True
        Whether to enable probability estimates.
    
    Examples
    --------
    >>> clf = SVMClassifier(kernel='linear', C=1.0)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        probability: bool = True
    ):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=probability,
            random_state=42
        )
    
    def fit(self, X, y) -> 'SVMClassifier':
        """Train the SVM model."""
        self.classes_ = np.unique(y)
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        if not self.probability:
            raise ValueError("Model was not trained with probability=True")
        return self.model.predict_proba(X)
    
    def tune_hyperparameters(
        self,
        X,
        y,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """Tune hyperparameters using GridSearchCV."""
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        
        y_encoded = self.label_encoder.fit_transform(y)
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_search.fit(X, y_encoded)
        
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.classes_ = np.unique(y)
        
        return grid_search.best_params_


class LogisticRegressionClassifier(BaseSentimentClassifier):
    """
    Logistic Regression Classifier for sentiment analysis.
    
    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=1000
        Maximum iterations for convergence.
    
    Examples
    --------
    >>> clf = LogisticRegressionClassifier(C=1.0)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            multi_class='multinomial'
        )
    
    def fit(self, X, y) -> 'LogisticRegressionClassifier':
        """Train the Logistic Regression model."""
        self.classes_ = np.unique(y)
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance based on coefficients.
        
        Parameters
        ----------
        feature_names : list
            List of feature names.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained.")
        
        # Average absolute coefficients across classes
        importance = np.mean(np.abs(self.model.coef_), axis=0)
        
        feature_importance = dict(zip(feature_names, importance))
        return dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def tune_hyperparameters(
        self,
        X,
        y,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """Tune hyperparameters using GridSearchCV."""
        if param_grid is None:
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
        
        y_encoded = self.label_encoder.fit_transform(y)
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_search.fit(X, y_encoded)
        
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.classes_ = np.unique(y)
        
        return grid_search.best_params_


class RandomForestClassifier(BaseSentimentClassifier):
    """
    Random Forest Classifier for sentiment analysis.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    Examples
    --------
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = SklearnRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X, y) -> 'RandomForestClassifier':
        """Train the Random Forest model."""
        self.classes_ = np.unique(y)
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20
    ) -> Dict[str, float]:
        """
        Get feature importance from the Random Forest model.
        
        Parameters
        ----------
        feature_names : list
            List of feature names.
        top_n : int, default=20
            Number of top features to return.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained.")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return dict(list(sorted_importance.items())[:top_n])
    
    def tune_hyperparameters(
        self,
        X,
        y,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """Tune hyperparameters using GridSearchCV."""
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        
        y_encoded = self.label_encoder.fit_transform(y)
        grid_search = GridSearchCV(
            SklearnRandomForest(random_state=42, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_search.fit(X, y_encoded)
        
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.classes_ = np.unique(y)
        
        return grid_search.best_params_


class LSTMClassifier:
    """
    LSTM Neural Network Classifier for sentiment analysis.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int, default=128
        Dimension of word embeddings.
    lstm_units : int, default=64
        Number of LSTM units.
    num_classes : int, default=3
        Number of output classes.
    max_len : int, default=100
        Maximum sequence length.
    dropout_rate : float, default=0.5
        Dropout rate for regularization.
    
    Examples
    --------
    >>> clf = LSTMClassifier(vocab_size=10000, num_classes=3)
    >>> clf.build_model()
    >>> clf.fit(X_train, y_train, epochs=10)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lstm_units: int = 64,
        num_classes: int = 3,
        max_len: int = 100,
        dropout_rate: float = 0.5
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.model = None
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.history = None
    
    def build_model(self, embedding_matrix: Optional[np.ndarray] = None) -> None:
        """
        Build the LSTM model architecture.
        
        Parameters
        ----------
        embedding_matrix : np.ndarray, optional
            Pre-trained embedding matrix.
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Embedding, LSTM, Dense, Dropout,
                Bidirectional, GlobalMaxPooling1D
            )
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model.")
        
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
                trainable=embedding_matrix is None
            ),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(
        self,
        X,
        y,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'LSTMClassifier':
        """
        Train the LSTM model.
        
        Parameters
        ----------
        X : array-like
            Training sequences.
        y : array-like
            Training labels.
        epochs : int, default=10
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        validation_split : float, default=0.2
            Fraction of data for validation.
        verbose : int, default=1
            Verbosity mode.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        if self.model is None:
            self.build_model()
        
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            self.history = self.model.fit(
                X, y_encoded,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=verbose
            )
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model.")
        
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        y_encoded = np.argmax(probabilities, axis=1)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath: str) -> None:
        """Save the model."""
        if self.model is not None:
            self.model.save(filepath)
            # Save label encoder separately
            joblib.dump(self.label_encoder, filepath + '_label_encoder.joblib')
            print(f"LSTM model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load a saved model."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath)
            self.label_encoder = joblib.load(filepath + '_label_encoder.joblib')
            self.is_fitted = True
            print(f"LSTM model loaded from {filepath}")
        except ImportError:
            raise ImportError("TensorFlow is required to load LSTM model.")


class BERTClassifier:
    """
    BERT-based Classifier for sentiment analysis.
    
    Uses DistilBERT for efficiency while maintaining good performance.
    
    Parameters
    ----------
    model_name : str, default='distilbert-base-uncased'
        Name of the pre-trained model.
    num_classes : int, default=3
        Number of output classes.
    max_length : int, default=128
        Maximum sequence length.
    
    Examples
    --------
    >>> clf = BERTClassifier(num_classes=3)
    >>> clf.fit(train_texts, train_labels, epochs=3)
    >>> predictions = clf.predict(test_texts)
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_classes: int = 3,
        max_length: int = 128
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.device = None
    
    def _setup_device(self):
        """Setup computing device (CPU/GPU)."""
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
        except ImportError:
            raise ImportError("PyTorch is required for BERT model.")
    
    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            from transformers import (
                DistilBertTokenizer,
                DistilBertForSequenceClassification
            )
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes
            )
            self.model.to(self.device)
        except ImportError:
            raise ImportError("Transformers library is required for BERT model.")
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'BERTClassifier':
        """
        Fine-tune the BERT model.
        
        Parameters
        ----------
        texts : list
            Training texts.
        labels : list
            Training labels.
        epochs : int, default=3
            Number of training epochs.
        batch_size : int, default=16
            Batch size for training.
        learning_rate : float, default=2e-5
            Learning rate for optimizer.
        validation_split : float, default=0.2
            Fraction of data for validation.
        verbose : bool, default=True
            Whether to print progress.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import AdamW
            from sklearn.model_selection import train_test_split
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(f"Required library not found: {e}")
        
        self._setup_device()
        self._load_model()
        
        # Encode labels
        self.label_encoder.fit(labels)
        y_encoded = self.label_encoder.transform(labels)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, y_encoded, test_size=validation_split, random_state=42
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}') if verbose else train_loader
            
            for batch in progress_bar:
                input_ids, attention_mask, batch_labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if verbose:
                    progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, batch_labels = [b.to(self.device) for b in batch]
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += len(batch_labels)
            
            val_accuracy = val_correct / val_total
            if verbose:
                print(f'Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, '
                      f'Val Accuracy: {val_accuracy:.4f}')
            
            self.model.train()
        
        self.is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        import torch
        
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        import torch
        import torch.nn.functional as F
        
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()
        
        return probabilities
    
    def save(self, filepath: str) -> None:
        """Save the model."""
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(filepath)
            self.tokenizer.save_pretrained(filepath)
            joblib.dump(self.label_encoder, f"{filepath}/label_encoder.joblib")
            print(f"BERT model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load a saved model."""
        try:
            from transformers import (
                DistilBertTokenizer,
                DistilBertForSequenceClassification
            )
            
            self._setup_device()
            self.tokenizer = DistilBertTokenizer.from_pretrained(filepath)
            self.model = DistilBertForSequenceClassification.from_pretrained(filepath)
            self.model.to(self.device)
            self.label_encoder = joblib.load(f"{filepath}/label_encoder.joblib")
            self.is_fitted = True
            print(f"BERT model loaded from {filepath}")
        except ImportError:
            raise ImportError("Transformers library is required to load BERT model.")


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample data
    texts = [
        "The equipment is working great",
        "Poor service quality",
        "Standard maintenance completed",
        "Equipment failure reported",
        "Excellent training session",
        "Safety protocols need improvement",
        "Normal inspection done",
        "Management is very supportive"
    ] * 10
    
    labels = ["positive", "negative", "neutral", "negative",
              "positive", "negative", "neutral", "positive"] * 10
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    print("=== Testing Classifiers ===\n")
    
    # Test Naive Bayes
    print("Naive Bayes:")
    nb_clf = NaiveBayesClassifier()
    nb_clf.fit(X_train, y_train)
    nb_predictions = nb_clf.predict(X_test)
    print(f"Sample predictions: {nb_predictions[:5]}")
    
    # Test SVM
    print("\nSVM:")
    svm_clf = SVMClassifier(kernel='linear')
    svm_clf.fit(X_train, y_train)
    svm_predictions = svm_clf.predict(X_test)
    print(f"Sample predictions: {svm_predictions[:5]}")
    
    # Test Logistic Regression
    print("\nLogistic Regression:")
    lr_clf = LogisticRegressionClassifier()
    lr_clf.fit(X_train, y_train)
    lr_predictions = lr_clf.predict(X_test)
    print(f"Sample predictions: {lr_predictions[:5]}")
    
    # Test Random Forest
    print("\nRandom Forest:")
    rf_clf = RandomForestClassifier(n_estimators=50)
    rf_clf.fit(X_train, y_train)
    rf_predictions = rf_clf.predict(X_test)
    print(f"Sample predictions: {rf_predictions[:5]}")
