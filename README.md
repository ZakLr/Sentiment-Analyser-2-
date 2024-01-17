# Emotion Classifier

This project utilizes Natural Language Processing techniques to classify emotions (fear, anger, joy) based on textual data. The classification is done using a Naive Bayes classifier trained on preprocessed text features.

## Overview

The project reads a CSV file containing textual data and corresponding emotion labels. It preprocesses the text using spaCy, tokenizes it, and vectorizes it using TF-IDF. The preprocessed text data is then used to train a Naive Bayes classifier. The trained model is saved for future use.


# Usage
Run the following command to train the model and save it:

```
python test.py
```
This will open a local server using Gradio library where you can test the model. In case you dont have gradio already installed run the following command:

```
pip install gradio
```

# File Descriptions
1.Emotion_classify_Data.csv: CSV file containing raw textual data and corresponding emotion labels.

2.main.py: Main script containing the code for preprocessing, model training, and saving the trained model.

3.emotion_detector_model.pkl: Saved Naive Bayes model after training.

4.test.py: a program to test the model in a web browser.

## Technologies Used
1.Python

2.pandas

3.scikit-learn

4.spaCy

5.joblib