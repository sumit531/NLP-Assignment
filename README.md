# NLP-Assignment
 NLP-Assignment
The provided Streamlit app is designed for predicting the similarity score between two input sentences using a pre-trained DistilBERT model. Here's a summary of the key functionalities:

Import Libraries:

Import necessary libraries, including Streamlit, pandas, NumPy, and TensorFlow.
Load Pre-trained Model and Tokenizer:

Load a pre-trained DistilBERT similarity model using TensorFlow.
Load the DistilBERT tokenizer for processing text.
Text Cleaning Function:

Define a function to clean input text by removing HTML tags, special characters, and digits.
Streamlit App Title:

Set the title of the Streamlit app as "Text Similarity Predictor."
Input Text Boxes for User:

Create two input text boxes for the user to enter two sentences.
Button to Calculate Similarity:

Include a button that, when clicked, triggers the calculation of similarity between the provided sentences.
Clean the Input Text:

Apply the defined text cleaning function to the input sentences.
Tokenize Input Text and Get DistilBERT Embeddings:

Tokenize the cleaned input sentences using the DistilBERT tokenizer.
Calculate DistilBERT embeddings for each sentence using a mean pooling approach.
Predict Similarity Score:

Utilize the pre-trained DistilBERT similarity model to predict the similarity score between the two sentences.
Display the Similarity Score:

Show the calculated similarity score to the user.
Streamlit App Execution:

The Streamlit app continuously runs, allowing users to input different sentence pairs and receive real-time similarity predictions.
In summary, the app provides an interactive interface for users to input sentences and receive predictions of semantic similarity based on the DistilBERT model. The model has been trained to capture semantic relationships between sentences, allowing users to explore the similarity between different textual inputs.





