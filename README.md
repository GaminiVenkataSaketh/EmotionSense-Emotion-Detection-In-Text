
# Emotion Detection in Text Using Natural Language Processing

![Home Page Screenshot](images/home%20page.png)


## Introduction

Emotion detection in text involves identifying the feelings expressed within written content. This is a complex task since emotions can be subtle and nuanced. Natural Language Processing (NLP) techniques enable automated analysis of text data to detect these emotions effectively.

The goal of this project is to build a model that leverages NLP to accurately classify emotions from textual data. Such a model can be applied in sentiment analysis, customer feedback evaluation, social media monitoring, and more.

---

## Dataset

The dataset used in this project consists of labeled text samples categorized into eight emotions: **anger, disgust, fear, joy, neutral, sadness, shame, and surprise**. It contains a total of **34,795 entries**, providing a diverse set of emotional expressions.

[Explore the dataset here](./data/)

---

## Methodology

The project follows these key steps:

1. **Data Preprocessing:**
   Text data is cleaned by removing stop words, punctuation, user mentions, and converting all text to lowercase to standardize inputs.

2. **Model Training:**
   Machine learning models, including Logistic Regression and Multinomial Naive Bayes, are trained on the processed text features to predict the corresponding emotions.

3. **Model Evaluation:**
   The performance of the models is assessed on a separate test set to measure their accuracy in emotion detection.

---

## Results

The Logistic Regression model achieved an accuracy of **62%** on the test dataset, demonstrating reasonable performance in emotion classification.

---

## Installation

To run this project locally, follow these steps:

1. Clone the repository to your machine:


2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. The app will automatically open in your default web browser.

---
