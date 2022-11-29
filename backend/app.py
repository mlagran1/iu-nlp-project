'''
Author: Morris LaGrand
Date: November, 2022

A Flask REST server for IU Intro to NLP final project.

Routes:
- /predict (POST): Predicts which office character said the quote
'''
###############################################################################
# Imports                                                                     #
###############################################################################
# System Imports
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import nltk
import re

# NTLK Prep
nltk.download('stopwords')
nltk.download('punkt')
stop_words = nltk.corpus.stopwords.words('english')

###############################################################################
# Globals                                                                     #
###############################################################################
MODELS_PATH = "../models"
CHARACTER_MAP = {
    0: "Dwight",
    1: "Jim",
    2: "Michael",
    3: "Pam"
}
# Load Models
CV = pickle.load(open(f"{MODELS_PATH}/cv.sav", 'rb'))
TT = pickle.load(open(f"{MODELS_PATH}/tt.sav", 'rb'))
SVC = pickle.load(open(f"{MODELS_PATH}/SVC.sav", 'rb'))

# Create Flask App
app = Flask(__name__)

###############################################################################
# Functions                                                                   #
###############################################################################
def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def predict(quote):
    # Preporcess quote
    proc_quote = normalize_document(quote)
    # Count Vectors
    cv_matrix = CV.transform(np.asarray([proc_quote])).toarray()
    # TF-IDF
    tt_matrix = TT.transform(cv_matrix).toarray()
    # SVC prediction
    prediction = SVC.predict(tt_matrix)

    return prediction[0]

###############################################################################
# Routes                                                                      #
###############################################################################
@app.route("/predict", methods=["POST"])
def predict_svc():
    '''
    Use trained SVC classifier to predict which Office character said the quote

    Parameters:
        N/A
    Returns:
        A dictionary object that contains the model's prediction
    '''
    data = request.get_json()
    # Get model key from post body
    quote = data["quote"]
    # Predict on quote
    pred = predict(quote)

    # Create response object
    response = {
        "prediction": CHARACTER_MAP[pred],
    }
    return response

###############################################################################
# Main                                                                        #
###############################################################################
def main():
    app.run(debug=False, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    main()
