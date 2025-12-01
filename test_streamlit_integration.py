#!/usr/bin/env python
"""
Full integration test simulating Streamlit app workflow
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import joblib
from src.preprocessing import TextPreprocessor
from src.model_registry import ModelRegistry

def load_models():
    """Load all trained models from the Model Registry (same as Streamlit)."""
    models = {}
    
    # Load model registry
    registry = ModelRegistry()
    all_models = registry.get_all_models()
    
    if not all_models:
        print("⚠️ No models found in registry.")
        return models
    
    # Load each registered model
    for display_name, model_info in all_models.items():
        model_path = model_info['model_path']
        vectorizer_path = model_info['vectorizer_path']
        
        # Check if files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                # Load model and vectorizer
                model_obj = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                models[display_name] = {
                    'model': model_obj,
                    'vectorizer': vectorizer,
                    'info': model_info
                }
            except Exception as e:
                print(f"⚠️ Could not load {display_name}: {e}")
    
    return models


def predict_sentiment(text, model_data, preprocessor):
    """Predict sentiment for a single text (same as Streamlit)."""
    # Preprocess
    processed_text = preprocessor.full_preprocess(text)
    
    # Get model and vectorizer from the model_data dictionary
    model_obj = model_data['model']
    vectorizer = model_data['vectorizer']

    # Transform
    X = vectorizer.transform([processed_text])
    
    # The model_obj is a dictionary saved by our custom classifiers
    if isinstance(model_obj, dict):
        # Extract the actual sklearn model and label encoder
        model = model_obj['model']
        label_encoder = model_obj['label_encoder']
        classes = label_encoder.classes_
        
        # Make prediction
        y_pred_encoded = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]
        
        # Get probabilities
        y_proba = model.predict_proba(X)[0]
    elif hasattr(model_obj, 'model'):
        # Custom wrapper class fallback
        model = model_obj.model
        label_encoder = model_obj.label_encoder
        classes = label_encoder.classes_
        
        y_pred_encoded = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]
        y_proba = model.predict_proba(X)[0]
    else:
        # Direct sklearn model fallback
        model = model_obj
        classes = model.classes_
        
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0]
    
    return y_pred, y_proba, classes


def test_streamlit_workflow():
    """Test the complete Streamlit workflow."""
    print("=" * 70)
    print("STREAMLIT APP INTEGRATION TEST")
    print("=" * 70)
    
    # Step 1: Load models (same as Streamlit)
    print("\n📥 Step 1: Loading models...")
    models = load_models()
    
    if not models:
        print("❌ No models loaded!")
        return False
    
    print(f"✅ Loaded {len(models)} models")
    for name in list(models.keys())[:3]:
        print(f"   - {name}")
    if len(models) > 3:
        print(f"   ... and {len(models) - 3} more")
    
    # Step 2: Initialize preprocessor (same as Streamlit)
    print("\n🔧 Step 2: Initializing preprocessor...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True
    )
    print("✅ Preprocessor ready")
    
    # Step 3: Test prediction with each model
    print("\n🧪 Step 3: Testing predictions...")
    print("-" * 70)
    
    test_text = "The new equipment is working great and has improved efficiency"
    
    success_count = 0
    for selected_model in models.keys():
        try:
            # This is exactly what Streamlit does
            prediction, probabilities, classes = predict_sentiment(
                test_text,
                models[selected_model],
                preprocessor
            )
            
            max_prob = max(probabilities) * 100
            success_count += 1
            
            # Only print first 3 for brevity
            if success_count <= 3:
                print(f"\n✅ {selected_model}")
                print(f"   Prediction: {prediction.upper()}")
                print(f"   Confidence: {max_prob:.1f}%")
            
        except Exception as e:
            print(f"\n❌ {selected_model}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if success_count > 3:
        print(f"\n... and {success_count - 3} more models tested successfully")
    
    print("\n" + "=" * 70)
    print(f"✅ ALL {success_count}/{len(models)} MODELS WORKING!")
    print("=" * 70)
    print("\n🎉 Streamlit app should work correctly now!")
    print("\n💡 Run: streamlit run app/streamlit_app.py")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_streamlit_workflow()
    sys.exit(0 if success else 1)

