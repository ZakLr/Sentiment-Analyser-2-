import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy

df = pd.read_csv('Emotion_classify_Data.csv')

X = df['Comment']  # Features/Predictors
y = df['Emotion'].map({"fear": 0, "anger": 1, "joy": 2})

# Preprocessing function using spaCy
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

X = X.apply(preprocess)

# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Vectorizing text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=7000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

# Naive Bayes model
NB_model = MultinomialNB()

# Model training
NB_model.fit(X_train_tfidf, y_train)

# Validation
preds = NB_model.predict(X_valid_tfidf)
accuracy = accuracy_score(y_valid, preds)
print("Accuracy:", accuracy)

# import joblib to save model
import joblib
joblib.dump(NB_model, 'emotion_detector_model.pkl')
joblib.dump(vectorizer, "vectorizer.pkl")

