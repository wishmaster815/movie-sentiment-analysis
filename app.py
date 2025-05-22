# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('rnn_model.h5')

# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 = OOV token, +3 for reserved indices
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit app
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')

# User input
user_input = st.text_area('✏️ Movie Review')

if st.button('🔍 Classify'):

    if not user_input.strip():
        st.warning("Please enter a non-empty review.")
    else:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = '✅ Positive' if prediction[0][0] > 0.5 else '❌ Negative'

        # Display the result
        st.success(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: `{prediction[0][0]:.4f}`')

else:
    st.info('Please enter a movie review to begin.')
