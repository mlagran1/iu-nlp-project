import streamlit as st
import requests
import os
import json
import pandas as pd
import random

# Globals
DATA = pd.read_csv("../data/test.csv")
QUOTES = list(zip(DATA["quote"], DATA["character"]))

FLASK_ADDRESS = os.environ.get("FLASK_ADDRESS", "http://0.0.0.0")
FLASK_PORT = os.environ.get("FLASK_PORT", "5001")

def main():
    app_mode = st.sidebar.selectbox('Select Page', ["Home", "Play"])

    if app_mode == "Home":
        st.title("Guess Who: The Office Edition")
        st.subheader(
        '''An NLP Classifier was trained on Office quotes to predict which character said the quote. Is your Office knowledge good enough to best the machine learning model? Play the game to find out!
        ''')
        st.image("../images/the-office.png")

    if app_mode == "Play":
        st.title("Guess Who: The Office Edition")
        img = st.image("../images/question-marks.png")
        # Sidebar
        st.sidebar.header("Who said it:")
        character = st.sidebar.radio('Character:', ("Dwight", "Jim", "Michael", "Pam"))
        guess = st.sidebar.button("Guess")
        next = st.sidebar.button("Next")

        # Quote
        if "quote" not in st.session_state:
            st.session_state["quote"] = random.choice(QUOTES)
        quote = st.session_state["quote"]
        st.info(quote[0])

        if guess:
            # Get Model's prediction
            resp = requests.post(f"{FLASK_ADDRESS}:{FLASK_PORT}/predict", json={"quote": quote[0]}).json()
            img.image(f"../images/{quote[1].lower()}.png")
            st.text(f"Correct Answer: {quote[1]}")
            st.text(f"You Guessed: {character}")
            st.text(f"The Model Guessed: {resp['prediction']}")

        if next:
            st.session_state["quote"] = random.choice(QUOTES)
            st.experimental_rerun()

if __name__ == '__main__':
    main()
