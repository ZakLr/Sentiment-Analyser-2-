# Emotion Classifier

This project utilizes Natural Language Processing techniques to classify emotions (fear, anger, joy) based on textual data. The classification is done using a Naive Bayes classifier trained on preprocessed text features.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [License](#license)

## Overview

The project reads a CSV file containing textual data and corresponding emotion labels. It preprocesses the text using spaCy, tokenizes it, and vectorizes it using TF-IDF. The preprocessed text data is then used to train a Naive Bayes classifier. The trained model is saved for future use.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ZakLr/emotion-classifier.git
   ```
Navigate to the project directory:
```
cd emotion-classifier
```
Install the required packages:
```
pip install -r requirements.txt
```
# Usage
Run the following command to train the model and save it:

```
python emotion_classifier.py
```
This will preprocess the data, train the Naive Bayes model, and save it as emotion_detector_model.pkl.

# File Descriptions
1.Emotion_classify_Data.csv: CSV file containing raw textual data and corresponding emotion labels.

2.emotion_classifier.py: Main script containing the code for preprocessing, model training, and saving the trained model.

3.emotion_detector_model.pkl: Saved Naive Bayes model after training.

4.Results: The model's accuracy on the validation set is printed to the console.

## Technologies Used
1.Python

2.pandas

3.scikit-learn

4.spaCy

5.joblib