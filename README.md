# Text Classification System – Spam Detection

## Overview
This project is a simple **machine learning based text classification system** that classifies SMS messages as **Spam** or **Ham (Not Spam)**.

The main goal of this project is to understand and implement the **complete machine learning pipeline** starting from raw text data to building a model and finally using it through a simple user interface.

This project was built as part of an **AI/ML Engineer Intern assignment**.

---

## Problem Statement
Given a raw SMS message as input, the system should predict whether the message is **Spam** or **Ham**.

Spam messages usually contain promotional or fraudulent content, while ham messages are normal personal messages.

This is a common **Natural Language Processing (NLP)** classification problem.

---

## Dataset
The dataset used is a publicly available **SMS Spam Dataset**.

- The dataset contains labeled SMS messages
- Labels:
  - `spam` → unwanted promotional messages
  - `ham` → normal messages
- The data is stored as a **plain text file** with tab-separated values

Dataset location in the project:
`data/raw/spam`

---

## Approach
The project follows a standard machine learning pipeline:

1. Load the raw text dataset
2. Preprocess and clean the text data
3. Convert text into numerical features using TF-IDF
4. Split the dataset into training and testing sets
5. Train machine learning models
6. Evaluate and compare model performance
7. Save the best performing model
8. Use the saved model in a simple web-based UI

---

## Text Preprocessing
Raw text data cannot be used directly for machine learning.  
Text preprocessing was performed to clean and normalize the data.

The following steps were applied:
- Convert text to lowercase
- Remove unnecessary characters and noise
- Remove common stopwords

These steps help reduce noise and improve model performance.

---

## Feature Extraction
Machine learning models work only with numerical data.

To convert text into numbers, **TF-IDF (Term Frequency–Inverse Document Frequency)** was used.  
TF-IDF assigns higher importance to meaningful words and lower importance to very common words.

This helps the model focus on words that are useful for classification.

---

## Models Used
Two different machine learning models were trained and compared:

### 1. Naive Bayes
- Probabilistic classifier
- Very effective for text classification tasks
- Fast and simple to train

### 2. Logistic Regression
- Linear classification model
- Learns a decision boundary between classes
- Used for comparison with Naive Bayes

Both models were trained using the same TF-IDF features.

---

## Results
The models were evaluated using accuracy and classification metrics on unseen test data.

- **Naive Bayes Accuracy:** ~97.9%
- **Logistic Regression Accuracy:** ~95.9%

Naive Bayes performed slightly better on this dataset and was selected as the final model.

---

## Model Saving
The best performing model (Naive Bayes) along with the TF-IDF vectorizer was saved to disk using `joblib`.

Saved model location:
`models/spam_classifier.pkl`
Saving the model allows it to be reused later without retraining.

---

## User Interface
A simple **web-based user interface** was built using **Streamlit**.

The UI allows users to:
- Enter an SMS message
- Click a button to classify the message
- Instantly see whether the message is **Spam** or **Ham**

This demonstrates how a trained machine learning model can be used in a real application.

---

## How to Run the Project

### 1. Install Dependencies
Make sure Python is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
From the project root directory:
```bash
streamlit run app/app.py
```
The application will open in your browser.
Enter an SMS message to see the prediction.
