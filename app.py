import streamlit as st
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import re

# Load the saved DistilBERT similarity model
loaded_model = tf.keras.models.load_model('distilbert_similarity_model')

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Text cleaning function using re
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Streamlit app
st.title('Text Similarity Predictor')

# Input text boxes for user to enter two sentences
text1 = st.text_input('Enter the first sentence:')
text2 = st.text_input('Enter the second sentence:')

# Button to calculate similarity
if st.button('Predict Similarity'):
    # Clean the input text
    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)

    # Tokenize input text and get DistilBERT embeddings
    def get_distilbert_embeddings(text):
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
        outputs = distilbert_model(inputs)
        return tf.reduce_mean(outputs.last_hidden_state, axis=1)  # Use mean pooling

    # Tokenize and get DistilBERT embeddings for text1 and text2
    X_text1 = get_distilbert_embeddings(cleaned_text1)
    X_text2 = get_distilbert_embeddings(cleaned_text2)

    # Reshape to (1, 768) for both text1 and text2
    X_text1 = tf.reshape(X_text1, (1, -1))
    X_text2 = tf.reshape(X_text2, (1, -1))

    # Predict similarity score
    similarity_score = loaded_model.predict([X_text1, X_text2])[0, 0]

    # Display the similarity score
    st.success(f'The predicted similarity score is: {similarity_score:.4f}')
