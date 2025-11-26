"""
Text Preprocessing Module

This module provides text preprocessing utilities for NLP tasks.
It includes functions for text cleaning, tokenization, stopword removal,
lemmatization, and a comprehensive TextPreprocessor class.

Author: Sentiment Analysis Team
"""

import re
import string
from typing import List, Optional, Union
import numpy as np

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data packages."""
    nltk_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    for package in nltk_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass


# Initialize NLTK data
download_nltk_data()


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks.
    
    This class provides methods for text cleaning, tokenization,
    stopword removal, lemmatization, and stemming.
    
    Parameters
    ----------
    lowercase : bool, default=True
        Whether to convert text to lowercase.
    remove_punctuation : bool, default=True
        Whether to remove punctuation.
    remove_numbers : bool, default=True
        Whether to remove numbers.
    remove_stopwords : bool, default=True
        Whether to remove stopwords.
    lemmatize : bool, default=True
        Whether to apply lemmatization.
    stem : bool, default=False
        Whether to apply stemming (mutually exclusive with lemmatize).
    min_word_length : int, default=2
        Minimum word length to keep.
    custom_stopwords : list, optional
        Additional custom stopwords to remove.
    
    Attributes
    ----------
    lemmatizer : WordNetLemmatizer
        NLTK lemmatizer instance.
    stemmer : PorterStemmer
        NLTK stemmer instance.
    stop_words : set
        Set of stopwords to remove.
    
    Examples
    --------
    >>> preprocessor = TextPreprocessor()
    >>> text = "The equipment is WORKING great! 123"
    >>> clean_text = preprocessor.full_preprocess(text)
    >>> print(clean_text)
    'equipment working great'
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords_flag = remove_stopwords
        self.lemmatize_flag = lemmatize
        self.stem_flag = stem
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra spaces, etc.
        
        Parameters
        ----------
        text : str
            Input text to clean.
        
        Returns
        -------
        str
            Cleaned text.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.clean_text("Hello!!!   World???")
        'hello world'
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Parameters
        ----------
        text : str
            Input text to tokenize.
        
        Returns
        -------
        list
            List of tokens.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.tokenize("Hello world!")
        ['hello', 'world']
        """
        if not isinstance(text, str):
            return []
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            download_nltk_data()
            tokens = word_tokenize(text)
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Parameters
        ----------
        tokens : list
            List of tokens.
        
        Returns
        -------
        list
            List of tokens with stopwords removed.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_stopwords(['the', 'equipment', 'is', 'working'])
        ['equipment', 'working']
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to a list of tokens.
        
        Parameters
        ----------
        tokens : list
            List of tokens.
        
        Returns
        -------
        list
            List of lemmatized tokens.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.lemmatize_tokens(['running', 'machines', 'better'])
        ['running', 'machine', 'better']
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to a list of tokens.
        
        Parameters
        ----------
        tokens : list
            List of tokens.
        
        Returns
        -------
        list
            List of stemmed tokens.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.stem_tokens(['running', 'machines', 'better'])
        ['run', 'machin', 'better']
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Parameters
        ----------
        tokens : list
            List of tokens.
        
        Returns
        -------
        list
            List of tokens meeting minimum length requirement.
        """
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def full_preprocess(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Apply full preprocessing pipeline to text.
        
        Parameters
        ----------
        text : str
            Input text to preprocess.
        return_tokens : bool, default=False
            If True, return list of tokens instead of joined string.
        
        Returns
        -------
        str or list
            Preprocessed text as string or list of tokens.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> text = "The NEW equipment is Working Great!!! 123"
        >>> preprocessor.full_preprocess(text)
        'new equipment working great'
        
        >>> preprocessor.full_preprocess(text, return_tokens=True)
        ['new', 'equipment', 'working', 'great']
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)
        
        # Step 3: Remove stopwords if specified
        if self.remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize or Stem
        if self.lemmatize_flag:
            tokens = self.lemmatize_tokens(tokens)
        elif self.stem_flag:
            tokens = self.stem_tokens(tokens)
        
        # Step 5: Filter by length
        tokens = self.filter_by_length(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_batch(
        self,
        texts: List[str],
        return_tokens: bool = False,
        show_progress: bool = False
    ) -> List[Union[str, List[str]]]:
        """
        Apply preprocessing to a batch of texts.
        
        Parameters
        ----------
        texts : list
            List of texts to preprocess.
        return_tokens : bool, default=False
            If True, return list of tokens instead of joined strings.
        show_progress : bool, default=False
            If True, show progress bar.
        
        Returns
        -------
        list
            List of preprocessed texts or token lists.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> texts = ["Hello world!", "Good morning!"]
        >>> preprocessor.preprocess_batch(texts)
        ['hello world', 'good morning']
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Preprocessing")
            except ImportError:
                pass
        
        return [self.full_preprocess(text, return_tokens) for text in texts]
    
    def get_preprocessing_summary(self, text: str) -> dict:
        """
        Get a summary of preprocessing steps applied to text.
        
        Parameters
        ----------
        text : str
            Input text.
        
        Returns
        -------
        dict
            Dictionary showing text at each preprocessing step.
        
        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> summary = preprocessor.get_preprocessing_summary("The equipment is GREAT!")
        >>> for step, result in summary.items():
        ...     print(f"{step}: {result}")
        """
        summary = {
            "original": text,
            "after_cleaning": self.clean_text(text),
        }
        
        tokens = self.tokenize(summary["after_cleaning"])
        summary["after_tokenization"] = tokens
        
        if self.remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)
            summary["after_stopword_removal"] = tokens
        
        if self.lemmatize_flag:
            tokens = self.lemmatize_tokens(tokens)
            summary["after_lemmatization"] = tokens
        elif self.stem_flag:
            tokens = self.stem_tokens(tokens)
            summary["after_stemming"] = tokens
        
        tokens = self.filter_by_length(tokens)
        summary["final_tokens"] = tokens
        summary["final_text"] = ' '.join(tokens)
        
        return summary


def preprocess_for_bert(text: str, max_length: int = 512) -> str:
    """
    Minimal preprocessing for BERT-based models.
    
    BERT has its own tokenizer, so we only do minimal cleaning.
    
    Parameters
    ----------
    text : str
        Input text.
    max_length : int, default=512
        Maximum text length (will be truncated).
    
    Returns
    -------
    str
        Minimally preprocessed text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if necessary
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def get_word_frequencies(
    texts: List[str],
    preprocessor: Optional[TextPreprocessor] = None,
    top_n: int = 50
) -> dict:
    """
    Get word frequencies from a list of texts.
    
    Parameters
    ----------
    texts : list
        List of texts.
    preprocessor : TextPreprocessor, optional
        Preprocessor to use. If None, creates a default one.
    top_n : int, default=50
        Number of top words to return.
    
    Returns
    -------
    dict
        Dictionary of word frequencies.
    
    Examples
    --------
    >>> texts = ["good equipment", "good service", "equipment maintenance"]
    >>> freqs = get_word_frequencies(texts, top_n=3)
    >>> print(freqs)
    {'good': 2, 'equipment': 2, 'service': 1}
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    word_counts = {}
    for text in texts:
        tokens = preprocessor.full_preprocess(text, return_tokens=True)
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
    
    # Sort by frequency and take top N
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_words[:top_n])


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "The new diagnostic tool has significantly improved our repair efficiency!",
        "Equipment keeps breaking down, we need better maintenance schedules...",
        "Completed the scheduled maintenance as per the manual.",
    ]
    
    print("=== Text Preprocessing Demo ===\n")
    
    for text in sample_texts:
        print(f"Original: {text}")
        summary = preprocessor.get_preprocessing_summary(text)
        print(f"Cleaned: {summary['after_cleaning']}")
        print(f"Tokens: {summary['after_tokenization']}")
        print(f"Final: {summary['final_text']}")
        print("-" * 50)
