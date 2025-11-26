"""
Feature Extraction Module

This module provides feature extraction utilities for text data,
including TF-IDF vectorization, Count vectorization, and word embeddings.

Author: Sentiment Analysis Team
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix


class TFIDFExtractor:
    """
    TF-IDF Feature Extractor wrapper class.
    
    This class wraps sklearn's TfidfVectorizer with additional
    functionality for sentiment analysis tasks.
    
    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of features to extract.
    ngram_range : tuple, default=(1, 2)
        Range of n-grams to extract.
    min_df : int, default=2
        Minimum document frequency for features.
    max_df : float, default=0.95
        Maximum document frequency for features.
    sublinear_tf : bool, default=True
        Apply sublinear TF scaling.
    
    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The underlying sklearn vectorizer.
    is_fitted : bool
        Whether the vectorizer has been fitted.
    feature_names_ : array
        Feature names after fitting.
    
    Examples
    --------
    >>> extractor = TFIDFExtractor(max_features=1000)
    >>> texts = ["good equipment", "bad service", "neutral feedback"]
    >>> features = extractor.fit_transform(texts)
    >>> print(features.shape)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf
        )
        
        self.is_fitted = False
        self.feature_names_ = None
    
    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        """
        Fit the vectorizer on the training texts.
        
        Parameters
        ----------
        texts : list
            List of training texts.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF features.
        
        Parameters
        ----------
        texts : list
            List of texts to transform.
        
        Returns
        -------
        csr_matrix
            Sparse matrix of TF-IDF features.
        
        Raises
        ------
        ValueError
            If the vectorizer has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit and transform texts in one step.
        
        Parameters
        ----------
        texts : list
            List of texts.
        
        Returns
        -------
        csr_matrix
            Sparse matrix of TF-IDF features.
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return features
    
    def get_feature_names(self) -> np.ndarray:
        """
        Get feature names.
        
        Returns
        -------
        array
            Array of feature names.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted.")
        return self.feature_names_
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top N features by average TF-IDF score.
        
        This requires the vectorizer to be fitted and have stored
        IDF values.
        
        Parameters
        ----------
        n : int, default=20
            Number of top features to return.
        
        Returns
        -------
        list
            List of top feature names.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted.")
        
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-n:][::-1]
        return [self.feature_names_[i] for i in top_indices]
    
    def save(self, filepath: str) -> None:
        """
        Save the vectorizer to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the vectorizer.
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"TF-IDF vectorizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a vectorizer from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the vectorizer from.
        """
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        print(f"TF-IDF vectorizer loaded from {filepath}")


class CountVectorizerExtractor:
    """
    Count Vectorizer Feature Extractor wrapper class.
    
    This class wraps sklearn's CountVectorizer with additional
    functionality for sentiment analysis tasks.
    
    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of features to extract.
    ngram_range : tuple, default=(1, 2)
        Range of n-grams to extract.
    min_df : int, default=2
        Minimum document frequency for features.
    max_df : float, default=0.95
        Maximum document frequency for features.
    binary : bool, default=False
        If True, all non-zero counts are set to 1.
    
    Attributes
    ----------
    vectorizer : CountVectorizer
        The underlying sklearn vectorizer.
    is_fitted : bool
        Whether the vectorizer has been fitted.
    feature_names_ : array
        Feature names after fitting.
    
    Examples
    --------
    >>> extractor = CountVectorizerExtractor(max_features=1000)
    >>> texts = ["good equipment", "bad service", "neutral feedback"]
    >>> features = extractor.fit_transform(texts)
    >>> print(features.shape)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        binary: bool = False
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary
        )
        
        self.is_fitted = False
        self.feature_names_ = None
    
    def fit(self, texts: List[str]) -> 'CountVectorizerExtractor':
        """
        Fit the vectorizer on the training texts.
        
        Parameters
        ----------
        texts : list
            List of training texts.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to count features.
        
        Parameters
        ----------
        texts : list
            List of texts to transform.
        
        Returns
        -------
        csr_matrix
            Sparse matrix of count features.
        
        Raises
        ------
        ValueError
            If the vectorizer has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit and transform texts in one step.
        
        Parameters
        ----------
        texts : list
            List of texts.
        
        Returns
        -------
        csr_matrix
            Sparse matrix of count features.
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return features
    
    def get_feature_names(self) -> np.ndarray:
        """
        Get feature names.
        
        Returns
        -------
        array
            Array of feature names.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted.")
        return self.feature_names_
    
    def get_vocabulary_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns
        -------
        int
            Number of features in vocabulary.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted.")
        return len(self.feature_names_)
    
    def save(self, filepath: str) -> None:
        """
        Save the vectorizer to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the vectorizer.
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"Count vectorizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a vectorizer from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the vectorizer from.
        """
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        print(f"Count vectorizer loaded from {filepath}")


def prepare_sequences_for_lstm(
    texts: List[str],
    max_words: int = 10000,
    max_len: int = 100,
    tokenizer=None
) -> Tuple[np.ndarray, any]:
    """
    Prepare text sequences for LSTM model.
    
    Parameters
    ----------
    texts : list
        List of texts.
    max_words : int, default=10000
        Maximum vocabulary size.
    max_len : int, default=100
        Maximum sequence length.
    tokenizer : Tokenizer, optional
        Pre-fitted tokenizer. If None, a new one will be created.
    
    Returns
    -------
    tuple
        (padded_sequences, tokenizer)
    """
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        raise ImportError("TensorFlow is required for LSTM sequence preparation.")
    
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    return padded_sequences, tokenizer


def get_word_embedding_matrix(
    tokenizer,
    embedding_dim: int = 100,
    embedding_path: Optional[str] = None
) -> np.ndarray:
    """
    Create word embedding matrix from pre-trained embeddings.
    
    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted Keras tokenizer.
    embedding_dim : int, default=100
        Embedding dimension.
    embedding_path : str, optional
        Path to pre-trained embeddings file (GloVe format).
        If None, returns random embeddings.
    
    Returns
    -------
    np.ndarray
        Embedding matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(tokenizer.word_index) + 1
    
    if embedding_path is None:
        # Return random embeddings
        return np.random.randn(vocab_size, embedding_dim) * 0.01
    
    # Load pre-trained embeddings
    embeddings_index = {}
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"Embedding file not found: {embedding_path}")
        return np.random.randn(vocab_size, embedding_dim) * 0.01
    
    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.randn(embedding_dim) * 0.01
    
    return embedding_matrix


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "The equipment is working great",
        "Poor service quality",
        "Standard maintenance completed",
        "Equipment failure reported",
        "Excellent training session"
    ]
    
    print("=== TF-IDF Feature Extraction Demo ===")
    tfidf = TFIDFExtractor(max_features=100, min_df=1)
    tfidf_features = tfidf.fit_transform(sample_texts)
    print(f"TF-IDF Features Shape: {tfidf_features.shape}")
    print(f"Feature Names (first 10): {list(tfidf.get_feature_names()[:10])}")
    
    print("\n=== Count Vectorizer Demo ===")
    count = CountVectorizerExtractor(max_features=100, min_df=1)
    count_features = count.fit_transform(sample_texts)
    print(f"Count Features Shape: {count_features.shape}")
    print(f"Vocabulary Size: {count.get_vocabulary_size()}")
