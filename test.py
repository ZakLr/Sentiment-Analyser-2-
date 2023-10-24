import joblib as jb
import spacy

model = jb.load('emotion_detector_model.pkl')
# input a text and preprocess it
nlp = spacy.load("en_core_web_sm")
while True:
    text = input("Enter your text: ")

    tokenized_text = nlp(text)
    tokens = [token.text for token in tokenized_text]
    preprocessed_text = " ".join(tokens)

    # Load the TfidfVectorizer used during training
    vectorizer = jb.load('vectorizer.pkl')  # Assuming you saved the vectorizer during training

    # Transform the input text using the loaded vectorizer
    input_features = vectorizer.transform([preprocessed_text])

    # Predict the emotion label
    predicted_emotion = model.predict(input_features)
    if (predicted_emotion ==0):
        print("fear")
    elif(predicted_emotion==1):
         print("anger")
    else:
         print("joy")