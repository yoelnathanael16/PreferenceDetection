import streamlit as st
import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

@st.cache_resource
def load_saved_objects():
    model = joblib.load('model_svm.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

svm_model, tfidf = load_saved_objects()

def preprocess_sentence(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(cleaned_tokens)

st.set_page_config(page_title="User Classifier")
st.title("User Preferences Topic Classification")
st.write("Enter your text in English below to detect your preference topic in real-time.")

user_input = st.text_area("Input Your Text:", height=150)

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text before clicking the button!")
    else:
        cleaned = preprocess_sentence(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = svm_model.predict(vectorized)[0]
        raw_scores = svm_model.decision_function(vectorized)[0]
        
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        confidence = max(probabilities) * 100
        
        if confidence < 45.0:
            st.warning(f"⚠️ **Prediction Result:** {prediction.upper()} *(Model is not confident: {confidence:.2f}%)*")
        else:
            st.success(f"🏆 **Prediction Result:** {prediction.upper()} *(Confidence: {confidence:.2f}%)*")
        
        st.write("### Class Probabilities (via SVM Softmax):")
        classes = svm_model.classes_
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        st.bar_chart(prob_dict)