#!/usr/bin/env python
"""
Test the predict_sentiment function with actual models
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import joblib
from src.preprocessing import TextPreprocessor
from src.model_registry import ModelRegistry

def test_prediction():
    """Test prediction with a real model."""
    print("=" * 60)
    print("Testing Prediction Function")
    print("=" * 60)
    
    # Load model registry
    registry = ModelRegistry()
    all_models = registry.get_all_models()
    
    if not all_models:
        print("❌ No models found!")
        return False
    
    # Pick the first model
    model_name = list(all_models.keys())[0]
    model_info = all_models[model_name]
    
    print(f"\n📦 Testing with: {model_name}")
    print(f"   Model Path: {model_info['model_path']}")
    print(f"   Vectorizer Path: {model_info['vectorizer_path']}")
    
    # Load model and vectorizer
    try:
        model_obj = joblib.load(model_info['model_path'])
        vectorizer = joblib.load(model_info['vectorizer_path'])
        print("   ✅ Loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return False
    
    # Check model structure
    print(f"\n🔍 Model structure:")
    print(f"   Type: {type(model_obj)}")
    if isinstance(model_obj, dict):
        print(f"   Keys: {list(model_obj.keys())}")
        print(f"   Inner model type: {type(model_obj.get('model', 'N/A'))}")
    
    # Create model_data dict (like in streamlit)
    model_data = {
        'model': model_obj,
        'vectorizer': vectorizer,
        'info': model_info
    }
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True
    )
    
    # Test texts
    test_texts = [
        "The new equipment is working great and has improved efficiency",
        "Equipment keeps breaking down, very frustrating",
        "Completed the maintenance as scheduled"
    ]
    
    print(f"\n🧪 Testing predictions:")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        try:
            # Preprocess
            processed_text = preprocessor.full_preprocess(text)
            
            # Get model and vectorizer
            model_obj = model_data['model']
            vectorizer = model_data['vectorizer']
            
            # Transform
            X = vectorizer.transform([processed_text])
            
            # The model_obj is a dictionary
            if isinstance(model_obj, dict):
                model = model_obj['model']
                label_encoder = model_obj['label_encoder']
                classes = label_encoder.classes_
                
                # Predict
                y_pred_encoded = model.predict(X)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]
                y_proba = model.predict_proba(X)[0]
            else:
                raise ValueError("Expected model_obj to be a dict")
            
            max_prob = max(y_proba) * 100
            
            print(f"\n{i}. Text: {text[:50]}...")
            print(f"   Prediction: {y_pred.upper()}")
            print(f"   Confidence: {max_prob:.1f}%")
            print(f"   Probabilities: {dict(zip(classes, y_proba))}")
            
        except Exception as e:
            print(f"\n{i}. ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✅ All predictions successful!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_prediction()
    sys.exit(0 if success else 1)

