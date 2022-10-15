# Library imports
import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from joblib import dump

# Load trained Pipeline
model = joblib.load('sentiment_model.pkl')

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)




# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']

    predictions = model.predict(new_review)[0]
    if predictions==0:
        return render_template('index.html', prediction_text='Negative Review 😢👎')
    else:
        return render_template('index.html', prediction_text='Positive Review 😃👍 ')


if __name__ == "__main__":
    app.run(debug=True)
