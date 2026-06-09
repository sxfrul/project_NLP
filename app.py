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

# Custom CSS for a flat, minimalist UI
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
st.markdown("Classify raw text into 20 distinct newsgroup categories using dynamic probability analysis.")

# --- INTUITIVE INPUT SECTION ---
st.subheader("Article Details")
col_input1, col_input2 = st.columns([1, 3])

with col_input1:
    subject_input = st.text_input("Subject Line (Optional)", placeholder="e.g. Next-gen ion thrusters")
with col_input2:
    content_input = st.text_area("Article Body", height=150, placeholder="Paste the full text of the article here...")

st.markdown("---")

# --- MODEL SELECTION ---
st.subheader("Select Models to Compare")
col1, col2 = st.columns(2)

with col1:
    model_1_choice = st.selectbox("Model 1", list(models.keys()), index=0)
with col2:
    model_2_choice = st.selectbox("Model 2", list(models.keys()), index=1)
    
if st.button("Categorize Text"):
    if content_input.strip() == "" and subject_input.strip() == "":
        st.warning("Please enter at least a subject or article body to classify.")
    else:
        with st.spinner("Analyzing text and generating dynamic visualizations..."):
            
            # Combine subject and content logically for the model
            combined_raw_text = f"Subject: {subject_input}\n\n{content_input}"
            cleaned_text = clean_text(combined_raw_text)
            
            # Helper function to get predictions and probabilities
            def get_inference(model_name):
                vec_name = "TF-IDF" if "TF-IDF" in model_name else "BoW"
                vec = vectorizers[vec_name]
                mod = models[model_name]
                
                vectorized_text = vec.transform([cleaned_text])
                
                # Get the actual predicted string
                prediction = mod.predict(vectorized_text)[0]
                
                # Get the probability array and match it to class names
                probs = mod.predict_proba(vectorized_text)[0]
                classes = mod.classes_
                
                return prediction, probs, classes

            pred_1, probs_1, classes_1 = get_inference(model_1_choice)
            pred_2, probs_2, classes_2 = get_inference(model_2_choice)
            
            # --- DISPLAY PREDICTIONS ---
            st.markdown("### Final Predictions")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**{model_1_choice}** predicts:\n### {pred_1}")
            with res_col2:
                st.info(f"**{model_2_choice}** predicts:\n### {pred_2}")

            st.markdown("---")
            st.markdown("### Dynamic Output Visualizations")
            
            # Process Data for Top 5 Probabilities
            def get_top_5_df(probs, classes):
                # Get indices of the top 5 probabilities
                top_5_idx = np.argsort(probs)[-5:][::-1]
                return pd.DataFrame({
                    'Category': classes[top_5_idx],
                    'Confidence (%)': probs[top_5_idx] * 100
                })

            df1_top5 = get_top_5_df(probs_1, classes_1)
            df2_top5 = get_top_5_df(probs_2, classes_2)

            vis_col1, vis_col2 = st.columns(2)
            
            # Visualization 1: Model 1 Probability Distribution
            with vis_col1:
                st.markdown(f"**1. {model_1_choice}: Top 5 Category Probabilities**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Confidence (%)", y="Category", data=df1_top5, palette="Blues_r", ax=ax1)
                ax1.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig1)

            # Visualization 2: Model 2 Probability Distribution
            with vis_col2:
                st.markdown(f"**2. {model_2_choice}: Top 5 Category Probabilities**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Confidence (%)", y="Category", data=df2_top5, palette="Greens_r", ax=ax2)
                ax2.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig2)

            # Visualization 3: Head-to-Head Confidence Comparison
            st.markdown("**3. Head-to-Head Top Choice Confidence Comparison**")
            
            # Get the confidence percentage of each model's top pick
            top_conf_1 = df1_top5.iloc[0]['Confidence (%)']
            top_conf_2 = df2_top5.iloc[0]['Confidence (%)']
            
            comparison_df = pd.DataFrame({
                'Model': [model_1_choice, model_2_choice],
                'Confidence on Primary Choice (%)': [top_conf_1, top_conf_2]
            })

            fig3, ax3 = plt.subplots(figsize=(10, 3))
            sns.barplot(x="Confidence on Primary Choice (%)", y="Model", data=comparison_df, palette="dark:gray", ax=ax3)
            ax3.set_xlim(0, 100)
            
            # Add text labels to the bars
            for index, value in enumerate(comparison_df['Confidence on Primary Choice (%)']):
                ax3.text(value + 1, index, f'{value:.1f}%', va='center')
                
            plt.tight_layout()
            st.pyplot(fig3)