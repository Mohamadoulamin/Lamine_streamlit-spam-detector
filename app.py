
import streamlit as st
import joblib

# Load the trained model and vectorizer
classifier = joblib.load('spam_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Spam Message Detector')

message = st.text_area('Enter a message:')

if st.button('Detect Spam'):
    if message:
        # Preprocess the message
        message_tfidf = tfidf_vectorizer.transform([message])
        
        # Predict
        prediction = classifier.predict(message_tfidf)[0]
        
        if prediction == 1:
            st.error('This is a SPAM message!')
        else:
            st.success('This is a HAM (legitimate) message.')
    else:
        st.warning('Please enter a message to detect.')


