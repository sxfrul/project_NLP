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
from deep_translator import GoogleTranslator
from wordcloud import WordCloud

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="News Categorizer", layout="wide")

# Minimalist UI: Strict square borders, solid colors, no gradients
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
        div[data-baseweb="tab-list"] {
            border-radius: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "subject" not in st.session_state:
    st.session_state.subject = ""
if "content" not in st.session_state:
    st.session_state.content = ""
if "translate" not in st.session_state:
    st.session_state.translate = False

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

# --- HELPER FUNCTIONS FOR ADVANCED VISUALS ---
def get_feature_importance(model, vectorizer, class_name, top_n=20):
    class_index = list(model.classes_).index(class_name)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract weights based on model type
    if hasattr(model, 'coef_'): # Logistic Regression
        importance = model.coef_[class_index]
    elif hasattr(model, 'feature_log_prob_'): # Naive Bayes
        importance = np.exp(model.feature_log_prob_[class_index])
    else:
        return pd.DataFrame()

    top_indices = np.argsort(importance)[-top_n:][::-1]
    return pd.DataFrame({
        'Word': feature_names[top_indices],
        'Weight': importance[top_indices]
    })

def generate_wordcloud(feature_df):
    word_freq = dict(zip(feature_df['Word'], feature_df['Weight']))
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# Hardcoded distribution of the 20 Newsgroups training set for rapid visualization
dataset_distribution = {
    'alt.atheism': 480, 'comp.graphics': 584, 'comp.os.ms-windows.misc': 591,
    'comp.sys.ibm.pc.hardware': 590, 'comp.sys.mac.hardware': 578, 'comp.windows.x': 593,
    'misc.forsale': 585, 'rec.autos': 594, 'rec.motorcycles': 598, 'rec.sport.baseball': 597,
    'rec.sport.hockey': 600, 'sci.crypt': 595, 'sci.electronics': 591, 'sci.med': 594,
    'sci.space': 593, 'soc.religion.christian': 599, 'talk.politics.guns': 546,
    'talk.politics.mideast': 564, 'talk.politics.misc': 465, 'talk.religion.misc': 377
}

# --- APP LAYOUT ---
st.title("News Article Categorization Engine")

# ==========================================
# STEP 1: ARTICLE INPUT
# ==========================================
if st.session_state.step == 1:
    st.markdown("Classify raw text into 20 distinct newsgroup categories using NLP pipelines.")
    st.subheader("Step 1: Article Details")
    
    subject_input = st.text_input("Subject Line", value=st.session_state.subject, placeholder="e.g. Next-gen ion thrusters")
    content_input = st.text_area("Article Body", height=200, value=st.session_state.content, placeholder="Paste the full text of the article here...")
    
    translate_toggle = st.checkbox("Auto-translate to English before processing", value=st.session_state.translate)

    st.markdown("<br>", unsafe_allow_html=True)
    
    warning_placeholder = st.empty()
    
    spacer, btn_col = st.columns([8.5, 1.5])
    with btn_col:
        next_clicked = st.button("Next ➔", use_container_width=True)
        
    if next_clicked:
        if subject_input.strip() == "" or content_input.strip() == "":
            warning_placeholder.warning("Please enter both a Subject Line and Article Body to proceed.")
        else:
            st.session_state.subject = subject_input
            st.session_state.content = content_input
            st.session_state.translate = translate_toggle
            st.session_state.step = 2
            st.rerun()

# ==========================================
# STEP 2: MODEL SELECTION & PREDICTION
# ==========================================
elif st.session_state.step == 2:
    st.subheader("Step 2: Select Models to Compare")
    
    with st.spinner("Preparing text environment..."):
        display_subject = st.session_state.subject
        display_content = st.session_state.content
        
        if st.session_state.translate:
            translator = GoogleTranslator(source='auto', target='en')
            display_subject = translator.translate(st.session_state.subject)
            display_content = translator.translate(st.session_state.content)
            st.caption(f"**Translated to English:** Subject: {display_subject} | Body: {display_content[:60]}...")
        else:
            st.caption(f"**Loaded Text:** Subject: {display_subject} | Body: {display_content[:60]}...")
    
    col1, col2 = st.columns(2)
    with col1:
        model_1_choice = st.selectbox("Model 1", list(models.keys()), index=0)
    with col2:
        model_2_choice = st.selectbox("Model 2", list(models.keys()), index=1)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn_back, spacer_mid, col_btn_predict = st.columns([1.5, 7, 1.5])
    
    with col_btn_back:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
            
    with col_btn_predict:
        predict_clicked = st.button("Categorize ✨", use_container_width=True)
        
    if predict_clicked:
        with st.spinner("Analyzing text and generating dynamic visualizations..."):
            
            combined_raw_text = f"Subject: {display_subject}\n\n{display_content}"
            cleaned_text = clean_text(combined_raw_text)
            
            def get_inference(model_name):
                vec_name = "TF-IDF" if "TF-IDF" in model_name else "BoW"
                vec = vectorizers[vec_name]
                mod = models[model_name]
                
                vectorized_text = vec.transform([cleaned_text])
                prediction = mod.predict(vectorized_text)[0]
                probs = mod.predict_proba(vectorized_text)[0]
                classes = mod.classes_
                return prediction, probs, classes, vec, mod

            pred_1, probs_1, classes_1, vec_1, mod_1 = get_inference(model_1_choice)
            pred_2, probs_2, classes_2, vec_2, mod_2 = get_inference(model_2_choice)
            
            st.markdown("---")
            st.markdown("### Final Predictions")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**{model_1_choice}** predicts:\n### {pred_1}")
            with res_col2:
                st.info(f"**{model_2_choice}** predicts:\n### {pred_2}")

            st.markdown("---")
            
            # Organize the heavy visualizations into tabs
            tab1, tab2, tab3 = st.tabs(["📊 Prediction Confidence", "☁️ Word Analysis", "📚 Dataset Insights"])
            
            # --- TAB 1: CONFIDENCE ---
            with tab1:
                def get_top_5_df(probs, classes):
                    top_5_idx = np.argsort(probs)[-5:][::-1]
                    return pd.DataFrame({
                        'Category': classes[top_5_idx],
                        'Confidence (%)': probs[top_5_idx] * 100
                    })

                df1_top5 = get_top_5_df(probs_1, classes_1)
                df2_top5 = get_top_5_df(probs_2, classes_2)

                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    st.markdown(f"**{model_1_choice}: Top 5 Probabilities**")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x="Confidence (%)", y="Category", data=df1_top5, palette="Blues_r", ax=ax1)
                    ax1.set_xlim(0, 100)
                    plt.tight_layout()
                    st.pyplot(fig1)

                with vis_col2:
                    st.markdown(f"**{model_2_choice}: Top 5 Probabilities**")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x="Confidence (%)", y="Category", data=df2_top5, palette="Greens_r", ax=ax2)
                    ax2.set_xlim(0, 100)
                    plt.tight_layout()
                    st.pyplot(fig2)

            # --- TAB 2: WORD ANALYSIS ---
            with tab2:
                st.markdown("Explore which specific words heavily influenced the model's final decision for its predicted category.")
                
                feat_col1, feat_col2 = st.columns(2)
                
                with feat_col1:
                    st.markdown(f"**Model 1 Word Importance ({pred_1})**")
                    df_feat_1 = get_feature_importance(mod_1, vec_1, pred_1, top_n=15)
                    fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x="Weight", y="Word", data=df_feat_1, palette="mako", ax=ax_f1)
                    plt.tight_layout()
                    st.pyplot(fig_f1)
                    
                    st.markdown("**Model 1 Word Cloud**")
                    df_wc_1 = get_feature_importance(mod_1, vec_1, pred_1, top_n=50)
                    st.pyplot(generate_wordcloud(df_wc_1))

                with feat_col2:
                    st.markdown(f"**Model 2 Word Importance ({pred_2})**")
                    df_feat_2 = get_feature_importance(mod_2, vec_2, pred_2, top_n=15)
                    fig_f2, ax_f2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x="Weight", y="Word", data=df_feat_2, palette="crest", ax=ax_f2)
                    plt.tight_layout()
                    st.pyplot(fig_f2)
                    
                    st.markdown("**Model 2 Word Cloud**")
                    df_wc_2 = get_feature_importance(mod_2, vec_2, pred_2, top_n=50)
                    st.pyplot(generate_wordcloud(df_wc_2))

            # --- TAB 3: DATASET DISTRIBUTION ---
            with tab3:
                st.markdown("**Original Training Data Distribution (20 Newsgroups)**")
                st.write("This chart visualizes the balanced nature of the foundational training dataset.")
                
                dist_df = pd.DataFrame(list(dataset_distribution.items()), columns=['Category', 'Document Count'])
                dist_df = dist_df.sort_values(by='Document Count', ascending=False)
                
                fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Category', y='Document Count', data=dist_df, palette="cubehelix", ax=ax_dist)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_dist)