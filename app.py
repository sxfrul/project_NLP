import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="News Categorizer", layout="wide")

# Custom CSS for a flat, minimalist UI (no gradients, square borders)
st.markdown("""
    <style>
        .stButton>button {
            border-radius: 0px !important;
            background: #2E2E2E !important;
            color: white !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 0px !important;
        }
        .css-1d391kg, .css-1n76uvr {
            border-radius: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- NLTK SETUP ---
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

download_nltk_data()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# --- LOAD MODELS & VECTORIZERS ---
@st.cache_resource
def load_assets():
    vectorizers = {
        "TF-IDF": joblib.load("tfidf_vectorizer.pkl"),
        "BoW": joblib.load("bow_vectorizer.pkl")
    }
    models = {
        "TF-IDF + LR": joblib.load("lr_tfidf_model.pkl"),
        "TF-IDF + NB": joblib.load("nb_tfidf_model.pkl"),
        "BoW + LR": joblib.load("lr_bow_model.pkl"),
        "BoW + NB": joblib.load("nb_bow_model.pkl")
    }
    return vectorizers, models

vectorizers, models = load_assets()

# --- APP LAYOUT ---
st.title("News Article Categorization Engine")
st.markdown("Classify raw text into 20 distinct newsgroup categories using NLP and Machine Learning.")

tab1, tab2 = st.tabs(["Prediction & Comparison", "Model Visualizations"])

# ==========================================
# TAB 1: PREDICTION AND COMPARISON
# ==========================================
with tab1:
    st.header("Live Text Inference")
    user_input = st.text_area("Paste article text or subject here:", height=150)
    
    st.subheader("Select Models to Compare")
    col1, col2 = st.columns(2)
    
    with col1:
        model_1_choice = st.selectbox("Model 1", list(models.keys()), index=0)
    with col2:
        model_2_choice = st.selectbox("Model 2", list(models.keys()), index=1)
        
    if st.button("Categorize Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Processing..."):
                cleaned_text = clean_text(user_input)
                
                # Helper function to predict
                def get_prediction(model_name):
                    vec_name = "TF-IDF" if "TF-IDF" in model_name else "BoW"
                    vec = vectorizers[vec_name]
                    mod = models[model_name]
                    vectorized_text = vec.transform([cleaned_text])
                    return mod.predict(vectorized_text)[0]
                
                pred_1 = get_prediction(model_1_choice)
                pred_2 = get_prediction(model_2_choice)
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"**{model_1_choice}** predicts:\n### {pred_1}")
                with res_col2:
                    st.info(f"**{model_2_choice}** predicts:\n### {pred_2}")

# ==========================================
# TAB 2: VISUALIZATIONS
# ==========================================
with tab2:
    st.header("Evaluation Metrics")
    
    # 1. Hardcoded Performance Data (matching your checkpoint)
    # Alternatively, you can read this from a CSV if you saved it in Colab
    perf_data = pd.DataFrame({
        "Model": ["TF-IDF + NB", "TF-IDF + LR", "BoW + NB", "BoW + LR"],
        "Accuracy": [0.83, 0.88, 0.81, 0.87], # Replace with your actual Colab output numbers
        "F1-Score": [0.82, 0.88, 0.80, 0.87]  # Replace with your actual Colab output numbers
    })
    
    vis_col1, vis_col2 = st.columns(2)
    
    # Visualization 1: Accuracy Comparison
    with vis_col1:
        st.subheader("1. Accuracy Comparison")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Accuracy", y="Model", data=perf_data, palette="Blues_d", ax=ax1)
        ax1.set_xlim(0.7, 1.0)
        st.pyplot(fig1)

    # Visualization 2: F1-Score Comparison
    with vis_col2:
        st.subheader("2. F1-Score Comparison")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x="F1-Score", y="Model", data=perf_data, palette="Greens_d", ax=ax2)
        ax2.set_xlim(0.7, 1.0)
        st.pyplot(fig2)

    # Visualization 3: Feature Matrix Sparsity (Visualizing the Vectorizers)
    st.subheader("3. Vocabulary Size (Max Features)")
    fig3, ax3 = plt.subplots(figsize=(8, 2))
    vocab_data = {"TF-IDF": 5000, "Bag of Words": 5000}
    ax3.barh(list(vocab_data.keys()), list(vocab_data.values()), color=['#2E2E2E', '#5E5E5E'])
    ax3.set_xlabel("Number of Features Extracted")
    st.pyplot(fig3)