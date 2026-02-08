
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load the saved model and vectorizer ---
# Ensure these files are in the same directory as app.py or provide the full path
try:
    log_reg_model = joblib.load('logistic_regression_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and TF-IDF vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or TF-IDF vectorizer files not found. Please ensure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are not found

# --- 2. Define preprocessing function and keywords ---
# Download NLTK data if not already downloaded (for Streamlit deployment)
# In a deployed Streamlit app, you might need to run these downloads once or include them in requirements.txt
# For local testing, ensure NLTK data is already present.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Assuming side_effect_keywords_lower was defined in the notebook context
# For the app, we need to re-define it or load it if it was saved.
# For simplicity, hardcoding it here as per common practice for app deployment
side_effect_keywords_lower = [
    'nausea', 'dizzy', 'headache', 'fatigue', 'insomnia', 'vomiting', 'pain', 
    'anxiety', 'depressed', 'rash', 'gain weight', 'lose weight', 'sweating', 
    'cramps', 'diarrhea', 'constipation', 'blurred vision', 'dry mouth', 
    'irritability', 'mood swings', 'sleep issues', 'stomach ache', 
    'heart palpitations', 'tremors', 'side effect', 'adverse reaction'
]

# --- 3. Streamlit app title ---
st.title("Drug Review Side Effect Predictor")
st.markdown("Enter a drug review below to predict if it mentions side effects.")

# --- 4. User input section ---
user_review = st.text_area("Enter drug review here:", height=150)

if st.button("Predict Side Effect"):
    if user_review:
        # Preprocess the user input
        cleaned_review = preprocess_text(user_review)
        
        # Transform the cleaned review using the loaded TF-IDF vectorizer
        # We need to ensure that the input to transform is a list of strings
        review_vectorized = tfidf_vectorizer.transform([cleaned_review])
        
        # Make prediction
        prediction = log_reg_model.predict(review_vectorized)
        prediction_proba = log_reg_model.predict_proba(review_vectorized)
        
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"Predicted: Side Effect (Confidence: {prediction_proba[0][1]*100:.2f}%) 🚨")
        else:
            st.success(f"Predicted: No Side Effect (Confidence: {prediction_proba[0][0]*100:.2f}%) ✅")
        
        st.markdown("---")
    else:
        st.warning("Please enter a review to get a prediction.")

# --- 5. Display model's overall performance metrics ---
st.subheader("Model Performance Metrics (from Test Set):")
# Hardcoding values obtained from previous evaluation step
accuracy = 0.9684
precision = 0.9880
recall = 0.9593
f1_score_val = 0.9735

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1-score:** {f1_score_val:.4f}")

# --- 6. Display confusion matrix ---
st.subheader("Confusion Matrix (from Test Set):")
# Hardcoding the confusion matrix from the previous evaluation step
cm = [[3758, 68],
      [237, 5593]]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Side Effect', 'Side Effect'],
            yticklabels=['No Side Effect', 'Side Effect'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix for Side Effect Prediction')
st.pyplot(fig)
