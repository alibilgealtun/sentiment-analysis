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
from src.utils import load_model


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
    """Load all trained models and vectorizer."""
    models = {}
    vectorizer = None
    
    model_files = {
        'Naive Bayes': 'models/naive_bayes_model.joblib',
        'SVM': 'models/svm_model.joblib',
        'Logistic Regression': 'models/logistic_regression_model.joblib',
        'Random Forest': 'models/random_forest_model.joblib'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[name] = load_model(filepath)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    
    if os.path.exists('models/tfidf_vectorizer.joblib'):
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    
    return models, vectorizer


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


def predict_sentiment(text, model_data, vectorizer, preprocessor):
    """Predict sentiment for a single text."""
    # Preprocess
    processed_text = preprocessor.full_preprocess(text)
    
    # Transform
    X = vectorizer.transform([processed_text])
    
    # Predict
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    y_pred_encoded = model.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]
    
    # Get probabilities
    y_proba = model.predict_proba(X)[0]
    
    return y_pred, y_proba, label_encoder.classes_


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
    models, vectorizer = load_models()
    preprocessor = load_preprocessor()
    
    # Check if models are loaded
    if not models:
        st.error("⚠️ No trained models found. Please run `python scripts/train_models.py` first.")
        st.stop()
    
    if vectorizer is None:
        st.error("⚠️ Vectorizer not found. Please run `python scripts/train_models.py` first.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=list(models.keys()).index('Logistic Regression') if 'Logistic Regression' in models else 0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Info")
    st.sidebar.info(f"**Selected Model:** {selected_model}")
    
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
                            vectorizer,
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
                                vectorizer,
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
        
        # Load results if available
        if os.path.exists('models/model_results.csv'):
            results_df = pd.read_csv('models/model_results.csv', index_col=0)
            
            st.dataframe(
                results_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            # Bar chart comparison
            fig = px.bar(
                results_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score'),
                x='index',
                y='Score',
                color='Metric',
                barmode='group',
                title='Model Comparison',
                labels={'index': 'Model'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results found. Train models first using `python scripts/train_models.py`")
    
    # Tab 4: Word Cloud
    with tab4:
        st.markdown('<h2 class="sub-header">Interactive Word Cloud</h2>', unsafe_allow_html=True)
        
        # Load dataset
        if os.path.exists('data/technician_feedback.csv'):
            df = pd.read_csv('data/technician_feedback.csv')
            
            sentiment_filter = st.selectbox(
                "Select Sentiment",
                options=['All', 'positive', 'negative', 'neutral']
            )
            
            if sentiment_filter == 'All':
                texts = df['feedback_text'].tolist()
                title = "Word Cloud - All Sentiments"
            else:
                texts = df[df['sentiment'] == sentiment_filter]['feedback_text'].tolist()
                title = f"Word Cloud - {sentiment_filter.title()} Sentiment"
            
            # Preprocess texts
            processed_texts = [preprocessor.full_preprocess(t) for t in texts]
            
            if processed_texts:
                fig = create_wordcloud(processed_texts, title)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No texts found for selected sentiment.")
        else:
            st.info("Dataset not found. Generate it using the data generator.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d;">
            <p>Built with ❤️ for Technician Feedback Analysis</p>
            <p>© 2024 Sentiment Analysis Team</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
