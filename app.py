import joblib
import string
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
tfidf = joblib.load('text_vector.pkl')
model = joblib.load('model.pkl')
def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("SpamClassifier")
input = st.text_area("Enter your text here")

btn = st.button("Predict")
if btn:
    text = transform_text(input)
    text_tr = tfidf.transform([text]).toarray()
    y_pred = model.predict(text_tr)
    if y_pred == 1:
        st.success("Spam Detected")
    else:
        st.error("Spam Not Detected")
