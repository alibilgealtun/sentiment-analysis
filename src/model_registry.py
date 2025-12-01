"""
Model Registry Module

Manages metadata for trained models including dataset information,
training timestamps, and performance metrics.

Author: Sentiment Analysis Team
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class ModelRegistry:
    """Registry to track trained models and their metadata."""

    def __init__(self, registry_path: str = 'models/model_registry.json'):
        """
        Initialize the model registry.

        Parameters
        ----------
        registry_path : str
            Path to the JSON file storing model metadata
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from file or create new one."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save registry to file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model_type: str,
        dataset_name: str,
        model_path: str,
        vectorizer_path: str,
        metrics: Dict,
        text_column: str,
        label_column: str,
        dataset_path: Optional[str] = None
    ):
        """
        Register a trained model with its metadata.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., 'SVM', 'Logistic Regression')
        dataset_name : str
            Name of the dataset used for training
        model_path : str
            Path to the saved model file
        vectorizer_path : str
            Path to the vectorizer file
        metrics : dict
            Performance metrics (accuracy, f1_score, etc.)
        text_column : str
            Name of text column used
        label_column : str
            Name of label column used
        dataset_path : str, optional
            Path to the dataset file
        """
        key = f"{model_type} - {dataset_name}"

        self.registry[key] = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'metrics': metrics,
            'text_column': text_column,
            'label_column': label_column,
            'dataset_path': dataset_path,
            'trained_at': datetime.now().isoformat(),
            'display_name': key
        }

        self._save_registry()
        print(f"✅ Registered model: {key}")

    def get_all_models(self) -> Dict:
        """Get all registered models."""
        return self.registry

    def get_model_info(self, display_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return self.registry.get(display_name)

    def get_models_by_type(self, model_type: str) -> Dict:
        """Get all models of a specific type."""
        return {
            k: v for k, v in self.registry.items()
            if v['model_type'] == model_type
        }

    def get_models_by_dataset(self, dataset_name: str) -> Dict:
        """Get all models trained on a specific dataset."""
        return {
            k: v for k, v in self.registry.items()
            if v['dataset_name'] == dataset_name
        }

    def list_model_names(self) -> List[str]:
        """Get list of all model display names."""
        return list(self.registry.keys())

    def remove_model(self, display_name: str):
        """Remove a model from the registry."""
        if display_name in self.registry:
            del self.registry[display_name]
            self._save_registry()
            print(f"✅ Removed model: {display_name}")
        else:
            print(f"⚠️  Model not found: {display_name}")

    def get_best_model(self, metric: str = 'f1_score') -> Optional[tuple]:
        """
        Get the best performing model based on a metric.

        Parameters
        ----------
        metric : str
            Metric to compare (default: 'f1_score')

        Returns
        -------
        tuple or None
            (display_name, model_info) of best model, or None if registry empty
        """
        if not self.registry:
            return None

        best = max(
            self.registry.items(),
            key=lambda x: x[1]['metrics'].get(metric, 0)
        )
        return best

    def print_summary(self):
        """Print a summary of all registered models."""
        if not self.registry:
            print("No models registered yet.")
            return

        print("\n" + "=" * 80)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 80)

        for display_name, info in self.registry.items():
            print(f"\n📦 {display_name}")
            print(f"   Model Type: {info['model_type']}")
            print(f"   Dataset: {info['dataset_name']}")
            print(f"   Accuracy: {info['metrics'].get('accuracy', 'N/A'):.4f}")
            print(f"   F1-Score: {info['metrics'].get('f1_score', 'N/A'):.4f}")
            print(f"   Trained: {info['trained_at']}")
            print(f"   Model Path: {info['model_path']}")

        print("\n" + "=" * 80)
        print(f"Total Models: {len(self.registry)}")
        print("=" * 80)


def get_registry() -> ModelRegistry:
    """Get or create the global model registry instance."""
    return ModelRegistry()

