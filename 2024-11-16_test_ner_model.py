import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from pathlib import Path
import json
import branca.colormap as cm
import plotly.graph_objects as go
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from wordcloud import WordCloud, get_single_color_func
import os


## Model Part

# Load the pre-trained model
@st.cache_data
# def load_model():
#     model = joblib.load("NER_model.joblib")
#     return model

# model = load_model()

def load_model():
    model_path = 'NER_model.joblib'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Model file {model_path} not found.")
        return None

model = load_model()

# Define stopwords
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5,
    }
    
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    
    return features

def parse(text):
    tokens = text.split()  # Tokenize the input text by space
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    
    # Make predictions using the model
    prediction = model.predict([features])[0]
    
    return tokens, prediction

# Add explanation mapping for predictions
def map_explanation(label):
    explanation = {
        "LOC": "Location (Tambon, Amphoe, Province)",
        "POST": "Postal Code",
        "ADDR": "Other Address Element",
        "O": "Not an Address"
    }
    return explanation.get(label, "Unknown")

# Set up the Streamlit app
# st.title("Try out the Named Entity Recognition (NER) model yourself!")
st.markdown(
    "<h1 style='font-size: 36px;'>Try out the Named Entity Recognition (NER) model yourself!</h1>",
    unsafe_allow_html=True
)

# Example input for NER analysis
example_input = "นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330"

# Text input for user data with the example as placeholder text
user_text = st.text_area("Enter any Thai Address below:", value="", placeholder=example_input)

# Button to make predictions
if st.button("Predict!"):
    # Make predictions
    tokens, predictions = parse(user_text)

    # Add explanations to predictions
    explanations = [map_explanation(pred) for pred in predictions]

    # Create a horizontal table
    data = pd.DataFrame([predictions, explanations], columns=tokens, index=["Prediction", "Explanation"])

    # Display the results
    st.write("Tokenized Results and Predictions with Explanations (Horizontal Table):")
    st.dataframe(data)

