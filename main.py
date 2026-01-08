import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

# load the word index
word_index=imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

model=load_model("05_RNN_PROJECT\model_imdb.h5")

# helper function
# function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?')for i in encoded_review])
## function to process user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review=[word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## prediction function
### prediction
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]> 0.5 else 'Negative'
    return sentiment,prediction[0][0]


## streamlit app

st.title("IMDB movie review sentiment analysis")
st.write("Enter a movie review to classsify it as positive or negative.")

# user input
user_input=st.text_area("Movie review")
if st.button("classify"):
    preprocessed_input=preprocess_text(user_input)
    
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]> 0.5 else 'Negative'
    
    # display the result
    st.write(f"sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review : ")