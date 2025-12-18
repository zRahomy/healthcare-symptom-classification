# Healthcare Symptom Classification  
**By:** Abdulalrahman Husham A Razzaq Alabbas  
**Student ID:** 1905983  

Multi-Class Disease Prediction Using Machine Learning & Deep Learning  

---

## Overview  
This project predicts a patient’s disease based on a short text describing their symptoms.  
It is a **30-class classification problem**, and the goal is to compare four different modeling approaches:

- TF-IDF + Logistic Regression  
- Feed-Forward Neural Network (FFNN)  
- Recurrent Neural Network (RNN)  
- Long Short-Term Memory Network (LSTM)  

All models follow the same preprocessing pipeline and the same dataset split to ensure a fair comparison.

---

## Dataset  
**Source:** Kaggle – Healthcare Symptoms–Disease Classification  
**Rows:** ~25,000  
**Features:**
- Age  
- Gender  
- Symptom text  
- Symptom count  
- Disease label (30 classes)  

The dataset is balanced, with ~800–900 samples per class.

---

## Preprocessing Steps  
The following preprocessing steps were applied consistently across all models:

### Numerical Features  
- Clean missing values  
- Encode gender numerically  
- Scale Age, Gender, and Symptom_Count with StandardScaler  

### Text Features  
- Convert text to lowercase  
- Remove punctuation and numbers  
- Remove stopwords  
- Apply stemming  
- Tokenize + pad sequences (NN models)  
- TF-IDF vectorization (TF-IDF model)  

A fixed split for FFNN, RNN and LSTM ensures identical train/test separation:  
- Train: 70%  
- Validation: 15%  
- Test: 15%  
- `random_state = 42`  
The TF-IDF split
- Train 85%
- Test 15%

---

## Model Architectures  

### 1. TF-IDF + Logistic Regression  
- TF-IDF vectorizer
- Multinomial Logistic Regression  
- Class weights balanced  

### 2. Feed-Forward Neural Network (FFNN)  
- Embedding layer  
- GlobalAveragePooling  
- Dense (128 → 64) with Dropout  
- Text + numeric features concatenated  
- Categorical Cross-Entropy + Adam  
- Early Stopping enabled  

### 3. Recurrent Neural Network (RNN)  
- Embedding  
- SimpleRNN(64)  
- Dense layers (128 → 64) + Dropout  
- Early stopping enabled  

### 4. Long Short-Term Memory Network (LSTM)  
- Embedding  
- LSTM(64)  
- Dense layers (128 → 64) + Dropout  
- Early stopping enabled  

---

## Final Model Results  

| Model | Test Accuracy |
|--------|---------------|
| TF-IDF + Logistic Regression | **3.6%** |
| Feed-Forward Neural Network (FFNN) | **3.6%** |
| Recurrent Neural Network (RNN) | **3.7% (Highest)** |
| LSTM | **3.5%** |

Random chance for 30 classes ≈ **3.3%**, meaning the models struggled to find strong patterns in the data.

---

## Why Accuracy Is Low  
- Many diseases share nearly identical symptoms  
- Input text is extremely short (2–6 words)  
- No medical context, time order, or symptom severity  
- Dataset likely synthetic or weakly correlated  
- Deep models cannot learn meaningful structure with weak text  
- Shows that model complexity does not solve weak data  

---

## How To Run The Project  
1. Clone the repository  
2. Install requirements  
3. Open notebooks in this order:
01 → 02 → 03 → 04 → 05 → 06
4. Train and evaluate each model  
5. Compare results  

