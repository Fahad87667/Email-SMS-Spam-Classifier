import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
nltk.download('punkt')


# Function to transform text
ps = PorterStemmer()
def transform_text(text):
    # Lower case
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = [i for i in text if i.isalnum()]

    # Removing stopwords
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in text]

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.image('spam.jpg', width=300)  
st.title('Email/SMS Spam Classifier')
st.text('Enter the message below:')
# Text input area
input_sms = st.text_area('')

# Prediction button
if st.button('Predict'):
    st.text('Processing... Transforming and classifying the message.')

    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input.reshape(1, -1))[0]

    # Display result with styles
    if result == 1:
        st.error('Spam Detected! 🚨')
    else:
        st.success('Not Spam! 👍')

# Footer
st.markdown("---")
