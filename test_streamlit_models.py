#!/usr/bin/env python
"""
Test script to verify Streamlit app can load models from registry
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import joblib
from src.model_registry import ModelRegistry

def test_model_loading():
    """Test loading models from the registry."""
    print("=" * 60)
    print("Testing Model Registry Loading")
    print("=" * 60)

    # Load model registry
    registry = ModelRegistry()
    all_models = registry.get_all_models()

    if not all_models:
        print("❌ No models found in registry!")
        return False

    print(f"\n✅ Found {len(all_models)} models in registry:\n")

    # Try to load each model
    loaded_count = 0
    for display_name, model_info in all_models.items():
        print(f"📦 {display_name}")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Dataset: {model_info['dataset_name']}")
        print(f"   Model Path: {model_info['model_path']}")
        print(f"   Vectorizer Path: {model_info['vectorizer_path']}")

        # Check if files exist
        model_exists = os.path.exists(model_info['model_path'])
        vectorizer_exists = os.path.exists(model_info['vectorizer_path'])

        print(f"   Model file exists: {model_exists}")
        print(f"   Vectorizer file exists: {vectorizer_exists}")

        if model_exists and vectorizer_exists:
            try:
                # Try to load
                model = joblib.load(model_info['model_path'])
                vectorizer = joblib.load(model_info['vectorizer_path'])
                print(f"   ✅ Successfully loaded!")
                loaded_count += 1
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
        else:
            print(f"   ⚠️  Files missing!")

        print()

    print("=" * 60)
    print(f"Summary: {loaded_count}/{len(all_models)} models loaded successfully")
    print("=" * 60)

    return loaded_count > 0


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)

