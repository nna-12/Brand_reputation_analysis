from joblib import load

# Load saved components
lr_model = load('bag_lr_model_lda.joblib')
lda = load('lda_transformer.joblib')
scaler = load('scaler.joblib')
label_encoder = load('label_encoder.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')


# Preprocess new text input (make sure to use the same preprocessing as before)
import re
import string

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = text.strip()
    return text

#X_new = ["good taste i love the coffee"]
X_new = ["Coffee here is so terrible"]
X_new_preprocessed = [preprocess_text(text) for text in X_new]

# Convert to TF-IDF using the original vectorizer
# Feature engineering
X_new_tfidf = tfidf_vectorizer.transform(X_new_preprocessed)

# Scale it
X_new_scaled = scaler.transform(X_new_tfidf.toarray())

# LDA transform
X_new_lda = lda.transform(X_new_scaled)

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
y_pred_encoded = lr_model.predict(X_new_lda)
print(X_new)
print(f"Predicted Sentiment (numeric): {y_pred_encoded[0]}")
print(f"Predicted Sentiment (label): {sentiment_map[y_pred_encoded[0]]}")
