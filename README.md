Healthcare Symptom Classification
By: Abdulalrahman Husham A Razzaq Alabbas
Student ID:1905983

Multi-Class Disease Prediction Using Machine Learning & Deep Learning

Overview

This project predicts a patient’s disease based on a short text describing their symptoms.
It is a 30-class classification problem, and the goal is to compare four different modeling approaches:

-TF-IDF + Logistic Regression

-Feed-Forward Neural Network (FFNN)

-Recurrent Neural Network (RNN)

-Long Short-Term Memory Network (LSTM)

All models follow the same preprocessing pipeline and the same dataset split to ensure a fair comparison.

Dataset

Source: Kaggle – Healthcare Symptoms–Disease Classification
Rows: ~25,000
Features:

-Age

-Gender

-Symptom text

-Symptom count

-Disease label (30 classes)

The dataset is balanced, with ~800–900 samples per class.

Preprocessing Steps

The following preprocessing steps were applied consistently across all models:

Numerical Features

-Clean missing values

-Encode gender numerically

-Scale Age, Gender, and Symptom_Count with StandardScaler

Text Features

-Convert text to lowercase

-Remove punctuation and numbers

-Remove stopwords

-Apply stemming

-Tokenize + pad sequences (NN models)

-TF-IDF vectorization (TF-IDF model)

-A fixed split ensures identical train/test separation:
Train: 75%
Validation: 15%
Test: 10%
random_state = 42

Model Architectures
1. TF-IDF + Logistic Regression

TF-IDF vectorizer (max_features=5000)

Multinomial Logistic Regression

Class weights balanced

2. Feed-Forward Neural Network (FFNN)

Embedding layer

GlobalAveragePooling

Dense (128 → 64) with Dropout

Text features + numeric features concatenated

Categorical Cross-Entropy + Adam

Early Stopping added

3. RNN

Embedding

SimpleRNN(64)

Dense layers (128 → 64) + Dropout

Early stopping enabled

4. LSTM

Embedding

LSTM(64)

Dense layers (128 → 64) + Dropout

Early stopping enabled

Final Model Results

All models performed close to random chance due to very short symptom descriptions and heavy symptom overlap across diseases.

Model	Test Accuracy
TF-IDF + Logistic Regression	2.9%
Feed-Forward Neural Network (FFNN)	3.6%
Recurrent Neural Network (RNN)	3.7% (highest)
LSTM	3.5%

Random chance for 30 classes ≈ 3.3%, meaning the models struggled to find strong patterns in the data.

Why Accuracy Is Low

-Many diseases share nearly identical symptoms

-Input text is extremely short (2–6 words)

-No medical context, temporal information, or severity indicators

-Dataset likely synthetic or weakly correlated

-Deep models cannot learn meaningful structure with such limited text

-The project demonstrates that model complexity does not solve weak data.

How to Run the Project

-Clone the repository

-Install requirements

-Open notebooks in order:

01 → 02 → 03 → 04 → 05 → 06


-Train and evaluate each model

-Compare results
