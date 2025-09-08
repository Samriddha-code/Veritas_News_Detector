# -*- coding: utf-8 -*-
"""
Veritas: Fake News Detector (Streamlit Web App)

This application provides a web interface to predict whether a news article
is real or fake using a pre-trained machine learning model.

To run this app:
1. Make sure you have the required libraries:
   pip install streamlit pandas numpy nltk scikit-learn
2. Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.
3. Run the following command in your terminal:
   streamlit run veritas_app.py
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import base64
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas | Fake News Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK resources if not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return True

download_nltk_data()

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_resources():
    """Load the pre-trained model and TF-IDF vectorizer from disk."""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        port_stem = PorterStemmer()
        return model, vectorizer, port_stem
    except FileNotFoundError:
        st.error("Error: 'model.pkl' or 'vectorizer.pkl' not found. Please ensure they are in the same directory.")
        return None, None, None

model, vectorizer, port_stem = load_resources()

# --- Text Preprocessing Function ---
def preprocess_text(content):
    """Preprocesses the input text in the same way as the training data."""
    if not port_stem or not content:
        return ""
    content = str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower()
    stemmed_words = [port_stem.stem(word) for word in stemmed_content.split() if word not in stopwords.words('english')]
    return ' '.join(stemmed_words)

# --- UI Rendering ---

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #1C2833;
    }
    .header-title {
        color: #EAF2F8;
        font-weight: bold;
    }
    .stButton > button {
        border: 2px solid #5DADE2;
        border-radius: 10px;
        color: #FFFFFF;
        background-color: #5DADE2;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #3498DB;
        background-color: #3498DB;
        color: #FFFFFF;
        transform: scale(1.02);
    }
    .stButton > button[kind="primary"] {
        background-color: #27AE60;
        border: 2px solid #27AE60;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #229954;
        border-color: #229954;
    }
    .stTextArea textarea {
        background-color: #2C3E50;
        border: 1px solid #566573;
        color: #ECF0F1;
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Header with New Logo and Title ---
# A new shield logo, Base64 encoded.
logo_data_base64 = "PHN2ZyB3aWR0aD0iNjRweCIgaGVpZ2h0PSI2NHB4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgZmlsbD0iIzM0OThkYiI+PHBhdGggZD0iTTEyIDJMMiA1djZjMCA1LjU1IDMuODQgMTAuNzQgOSAxMiA1LjE2LTEuMjYgOS02LjQ1IDktMTJWNUwxMiAyeiBtLTEuMDYgMTQuNDRsLTMuNTQtMy41NCAxLjQxLTEuNDEgMi4xMiAyLjEyIDQuMjQtNC4yNCAxLjQxIDEuNDEtNS42NSA1LjY2eiIvPjwvc3ZnPg=="

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
        <img src="data:image/svg+xml;base64,{logo_data_base64}" width="60">
        <h1 class="header-title" style="margin-left: 10px;">Veritas</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Fake News Detector")
st.write("Paste the news article text below to check if it's real or fake.")


# --- Session State Initialization ---
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --- Input Text Area ---
user_text = st.text_area(
    "Article Text",
    value=st.session_state.user_input,
    height=250,
    placeholder="Enter the full text of the news article here...",
    key='user_input_widget'
)
st.session_state.user_input = user_text

# --- Buttons: Predict and Clear ---
col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("Analyze News", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear & Reset", use_container_width=True)

# --- Logic for Buttons ---
if clear_button:
    st.session_state.user_input = ""
    st.session_state.prediction_result = None
    st.rerun()

if predict_button and model and vectorizer:
    if not st.session_state.user_input.strip():
        st.warning("Please enter some text to analyze.")
        st.session_state.prediction_result = None
    else:
        with st.spinner('Analyzing the text...'):
            processed_text = preprocess_text(st.session_state.user_input)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)
            st.session_state.prediction_result = prediction[0]

# --- Display Result ---
if st.session_state.prediction_result is not None:
    st.markdown("---")
    st.subheader("Analysis Result")
    if st.session_state.prediction_result == 1:
        st.success("✅ This appears to be REAL NEWS.")
    else:
        st.error("❌ This appears to be FAKE NEWS.")

# --- Footer Removed ---
# The "Built with Streamlit" footer has been removed.

