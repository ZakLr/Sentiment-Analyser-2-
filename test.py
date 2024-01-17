import joblib as jb
import spacy

model = jb.load('emotion_detector_model.pkl')
# input a text and preprocess it
nlp = spacy.load("en_core_web_sm")
import gradio as gr
def predict_val(text):
    tokenized_text = nlp(text)
    tokens = [token.text for token in tokenized_text]
    preprocessed_text = " ".join(tokens)

    # Load the TfidfVectorizer used during training
    vectorizer = jb.load('vectorizer.pkl')  # Assuming you saved the vectorizer during training

    # Transform the input text using the loaded vectorizer
    input_features = vectorizer.transform([preprocessed_text])

    # Predict the emotion label
    predicted_emotion = model.predict(input_features)
    return "fear/sad" if predicted_emotion == 0 else "anger" if predicted_emotion == 1 else "joy"


iface = gr.Interface(
 fn=predict_val,
 inputs="text",
 outputs="text",
 live=True,
 title="Spam Detector",
 description=""
)


iface.launch()