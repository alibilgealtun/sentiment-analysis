"""
Streamlit Web Application for Sentiment Analysis

This application provides an interactive interface for:
- Real-time sentiment prediction
- Model selection and comparison
- Batch prediction from CSV files
- Visualization of prediction confidence

Usage:
    streamlit run app/streamlit_app.py

Author: Sentiment Analysis Team
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

# Import project modules
from src.preprocessing import TextPreprocessor
from src.model_registry import ModelRegistry


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis - Technician Feedback",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #3498db; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models from the Model Registry."""
    models = {}

    # Load model registry
    registry = ModelRegistry()
    all_models = registry.get_all_models()

    if not all_models:
        st.warning("No models found in registry. Please train models first.")
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
                st.warning(f"Could not load {display_name}: {e}")
        else:
            missing = []
            if not os.path.exists(model_path):
                missing.append(f"model ({model_path})")
            if not os.path.exists(vectorizer_path):
                missing.append(f"vectorizer ({vectorizer_path})")
            st.warning(f"Files not found for {display_name}: {', '.join(missing)}")

    return models


@st.cache_resource
def load_preprocessor():
    """Load text preprocessor."""
    return TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True
    )


def predict_sentiment(text, model_data, preprocessor):
    """Predict sentiment for a single text."""
    # Preprocess
    processed_text = preprocessor.full_preprocess(text)
    
    # Get model and vectorizer from the model_data dictionary
    model_obj = model_data['model']
    vectorizer = model_data['vectorizer']

    # Transform
    X = vectorizer.transform([processed_text])
    
    # The model_obj is a dictionary saved by our custom classifiers
    # It has keys: 'model', 'label_encoder', 'classes_', 'is_fitted'
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
        # Custom wrapper class (e.g., from src.models) - fallback
        model = model_obj.model
        label_encoder = model_obj.label_encoder
        classes = label_encoder.classes_
        
        y_pred_encoded = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]
        y_proba = model.predict_proba(X)[0]
    else:
        # Direct sklearn model - fallback
        model = model_obj
        classes = model.classes_
        
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0]
    
    return y_pred, y_proba, classes


def create_probability_chart(probabilities, classes):
    """Create a bar chart for prediction probabilities."""
    # Define colors for each sentiment
    colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#3498db'
    }
    
    color_list = [colors.get(c, '#95a5a6') for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities * 100,
            marker_color=color_list,
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence',
        xaxis_title='Sentiment',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        showlegend=False,
        height=300
    )
    
    return fig


def create_wordcloud(texts, title="Word Cloud"):
    """Create a word cloud from texts."""
    combined_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Technician Feedback Classification System</p>', unsafe_allow_html=True)
    
    # Load resources
    models = load_models()
    preprocessor = load_preprocessor()
    
    # Check if models are loaded
    if not models:
        st.error("⚠️ No trained models found. Please run `python run_training.py` first.")
        st.info("💡 To train models, run: `python run_training.py --dataset your_data.csv`")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model Selection
    st.sidebar.markdown("### 🤖 Model Selection")
    st.sidebar.info(f"Found {len(models)} trained models")

    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        options=list(models.keys()),
        help="Select a trained model from the registry"
    )
    
    # Display model information
    if selected_model:
        model_info = models[selected_model]['info']
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Model Info")
        st.sidebar.markdown(f"**Type:** {model_info['model_type']}")
        st.sidebar.markdown(f"**Dataset:** {model_info['dataset_name']}")
        st.sidebar.markdown(f"**Accuracy:** {model_info['metrics']['accuracy']:.4f}")
        st.sidebar.markdown(f"**F1-Score:** {model_info['metrics']['f1_score']:.4f}")

        # Show training date
        trained_at = model_info.get('trained_at', 'Unknown')
        if trained_at != 'Unknown':
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(trained_at)
                st.sidebar.markdown(f"**Trained:** {dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.sidebar.markdown(f"**Trained:** {trained_at}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Prediction",
        "📁 Batch Prediction",
        "📊 Model Performance",
        "☁️ Word Cloud"
    ])
    
    # Tab 1: Single Prediction
    with tab1:
        st.markdown('<h2 class="sub-header">Real-time Sentiment Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter technician feedback:",
                height=150,
                placeholder="e.g., The new diagnostic tool has significantly improved our repair efficiency"
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        prediction, probabilities, classes = predict_sentiment(
                            text_input,
                            models[selected_model],
                            preprocessor
                        )
                    
                    # Display result
                    sentiment_colors = {
                        'positive': '🟢',
                        'negative': '🔴',
                        'neutral': '🔵'
                    }
                    
                    st.markdown("---")
                    st.markdown(f"### Result: {sentiment_colors.get(prediction, '⚪')} {prediction.upper()}")
                    
                    # Probability chart
                    fig = create_probability_chart(probabilities, classes)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.markdown("### 💡 Example Inputs")
            
            examples = [
                ("Positive", "The new equipment is working flawlessly and has improved our efficiency"),
                ("Negative", "Equipment keeps breaking down, we need better maintenance schedules"),
                ("Neutral", "Completed the scheduled maintenance as per the manual")
            ]
            
            for label, example in examples:
                with st.expander(f"{label} Example"):
                    st.write(example)
                    if st.button(f"Use this example", key=f"example_{label}"):
                        st.session_state.text_input = example
    
    # Tab 2: Batch Prediction
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Prediction</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with a 'text' or 'feedback_text' column",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Find text column
            text_col = None
            for col in ['text', 'feedback_text', 'Text', 'Feedback']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("CSV must contain a 'text' or 'feedback_text' column")
            else:
                st.write(f"Found {len(df)} rows")
                st.dataframe(df.head())
                
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing..."):
                        predictions = []
                        confidences = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df[text_col]):
                            pred, proba, classes = predict_sentiment(
                                str(text),
                                models[selected_model],
                                preprocessor
                            )
                            predictions.append(pred)
                            confidences.append(max(proba) * 100)
                            progress_bar.progress((i + 1) / len(df))
                        
                        df['predicted_sentiment'] = predictions
                        df['confidence'] = confidences
                    
                    st.success("✅ Batch prediction complete!")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Distribution chart
                    fig = px.pie(
                        df,
                        names='predicted_sentiment',
                        title='Predicted Sentiment Distribution',
                        color='predicted_sentiment',
                        color_discrete_map={
                            'positive': '#2ecc71',
                            'negative': '#e74c3c',
                            'neutral': '#3498db'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Performance
    with tab3:
        st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Create performance dataframe from registry
        if models:
            # Extract metrics from all loaded models
            performance_data = []
            for display_name, model_data in models.items():
                info = model_data['info']
                metrics = info['metrics']
                performance_data.append({
                    'Model': display_name,
                    'Type': info['model_type'],
                    'Dataset': info['dataset_name'],
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0)
                })

            perf_df = pd.DataFrame(performance_data)

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                dataset_filter = st.selectbox(
                    'Filter by Dataset:',
                    ['All'] + sorted(perf_df['Dataset'].unique().tolist())
                )
            with col2:
                metric_sort = st.selectbox(
                    'Sort by:',
                    ['F1-Score', 'Accuracy', 'Precision', 'Recall']
                )

            # Apply filter
            if dataset_filter != 'All':
                display_df = perf_df[perf_df['Dataset'] == dataset_filter].copy()
            else:
                display_df = perf_df.copy()

            # Sort
            display_df = display_df.sort_values(by=metric_sort, ascending=False)

            # Display table
            st.markdown("### 📊 Performance Comparison")
            st.dataframe(
                display_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}'
                }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            st.markdown("### 📈 Visual Comparison")

            # Create comparison chart
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig = go.Figure()

            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=display_df['Model'],
                    y=display_df[metric],
                    text=display_df[metric].apply(lambda x: f'{x:.3f}'),
                    textposition='auto'
                ))

            fig.update_layout(
                barmode='group',
                title=f'Model Performance Comparison{" - " + dataset_filter if dataset_filter != "All" else ""}',
                xaxis_title='Model',
                yaxis_title='Score',
                yaxis_range=[0, 1],
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Best model highlight
            best_model_idx = display_df[metric_sort].idxmax()
            best_model = display_df.loc[best_model_idx]

            st.markdown("### 🏆 Best Model")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", best_model['Type'])
            with col2:
                st.metric("Dataset", best_model['Dataset'])
            with col3:
                st.metric(metric_sort, f"{best_model[metric_sort]:.4f}")
            with col4:
                st.metric("Overall F1", f"{best_model['F1-Score']:.4f}")

        else:
            st.info("No model performance data available. Train models first using `python run_training.py`")

    # Tab 4: Word Cloud
    with tab4:
        st.markdown('<h2 class="sub-header">Interactive Word Cloud</h2>', unsafe_allow_html=True)
        st.info("Upload a CSV file to generate word clouds from your data")

        # Upload dataset for word cloud
        wordcloud_file = st.file_uploader(
            "Upload CSV file for word cloud visualization:",
            type=['csv'],
            key='wordcloud_upload'
        )

        if wordcloud_file is not None:
            df_wordcloud = pd.read_csv(wordcloud_file)

            # Auto-detect text and sentiment columns
            text_col = None
            sentiment_col = None

            for col in ['feedback_text', 'text', 'Text', 'comment', 'review']:
                if col in df_wordcloud.columns:
                    text_col = col
                    break

            for col in ['sentiment', 'label', 'Sentiment', 'Label', 'class']:
                if col in df_wordcloud.columns:
                    sentiment_col = col
                    break

            if text_col and sentiment_col:
                # Get unique sentiments
                unique_sentiments = df_wordcloud[sentiment_col].unique().tolist()
                sentiment_options = ['All'] + unique_sentiments

                sentiment_filter = st.selectbox(
                    "Select Sentiment",
                    options=sentiment_options
                )

                if sentiment_filter == 'All':
                    texts = df_wordcloud[text_col].tolist()
                    title = "Word Cloud - All Sentiments"
                else:
                    texts = df_wordcloud[df_wordcloud[sentiment_col] == sentiment_filter][text_col].tolist()
                    title = f"Word Cloud - {sentiment_filter.title()} Sentiment"

                # Preprocess texts
                processed_texts = [preprocessor.full_preprocess(t) for t in texts if pd.notna(t)]

                if processed_texts:
                    fig = create_wordcloud(processed_texts, title)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No texts found for selected sentiment.")
            elif text_col:
                st.info(f"Found text column '{text_col}' but no sentiment column. Showing all text.")
                texts = df_wordcloud[text_col].tolist()
                processed_texts = [preprocessor.full_preprocess(t) for t in texts if pd.notna(t)]

                if processed_texts:
                    fig = create_wordcloud(processed_texts, "Word Cloud")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("Could not find text column. Please ensure your dataset has a column named 'feedback_text', 'text', or similar.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d;">
            <p>Built by Group 13</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
