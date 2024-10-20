# Note: To run this code write streamlit run app.py on the terminal 

import streamlit as st
import pickle as pkl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import string

# Load pre-trained model and vectorizer
tfidf = pkl.load(open("Vectorizer.pkl", "rb"))
model = pkl.load(open("Model.pkl", "rb"))

# Set up the Streamlit app header
st.header("Email/SMS Spam Classifier")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess the input text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove mentions (e.g., @username)
    text = re.sub(r"@[\w-]+", "", text)
    
    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Strip unusual whitespaces
    text = text.strip()
    
    # Tokenize words
    text = word_tokenize(text)
    
    # Remove stop words
    text = [word for word in text if word not in stopwords.words("english")]

    # Part-of-speech tagging
    tagged_text = pos_tag(text)
    
    # Lemmatize words based on their POS tags
    lemmatized_text = [
        lemmatizer.lemmatize(word, pos=tag[0].lower()[0]) 
        if tag[0].lower()[0] in ['a', 'n', 'v'] 
        else lemmatizer.lemmatize(word)
        for word, tag in tagged_text
    ]

    # Return the cleaned and lemmatized text as a single string
    return " ".join(lemmatized_text)

# Function to validate input text
def validate_input(text):
    # Check if the input is empty
    if not text:
        return "Review cannot be empty."
    # Check if the input length is at least 10 characters
    if len(text) < 10:
        return "Review must be at least 10 characters long."

# Create a text area for user input
input_text = st.text_area("Enter your SMS/Email")

# Button to submit the input for classification
if st.button("Predict"):
    # Validate the input text
    validate = validate_input(input_text)
    if validate:
        # If validation fails, show an error message
        st.error(validate)
    else:
        # Clean and preprocess the text
        text = clean_text(input_text)
        # Transform the text using the TF-IDF vectorizer
        vectorized_text = tfidf.transform([text])
        # Make a prediction using the trained model
        prediction = model.predict(vectorized_text)[0]
        # Display the result based on the prediction
        if prediction == 0:
            st.write("Not Spam")
        else:
            st.write("Spam")
else:
    # Prompt the user to submit text for classification
    st.write("Click the submit button to classify the text")
