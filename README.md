# 🧠 SQL Injection Detection using Machine Learning Approaches

A machine learning–based system designed to **detect SQL Injection (SQLi) attacks** from user inputs and web requests.  
The project applies supervised learning and deep learning models to classify whether given input data is malicious (SQLi) or safe, strengthening web application security.

---

## 🚀 Features

- Detects SQL Injection payloads using ML classification  
- Compares multiple algorithms to identify the most accurate model  
- Employs **Count Vectorization** for text feature extraction  
- Uses labeled datasets from Kaggle containing SQL and non-SQL sentences  
- Evaluates performance using accuracy, precision, recall, and F1-score  
- Demonstrates a working example where user input is classified in real time  

---

## 🧩 Machine Learning Models Used

| Category | Model | Description |
|-----------|--------|-------------|
| **Probabilistic Model** | **Naïve Bayes** | Uses Bayes’ theorem to classify input text as SQL or non-SQL based on feature likelihoods. |
| **Instance-Based Model** | **K-Nearest Neighbors (KNN)** | Classifies input text by measuring similarity (Euclidean distance) to known labeled data. |
| **Tree-Based Model** | **Decision Tree** | Splits data hierarchically to decide whether input patterns indicate SQL injection. |
| **Margin-Based Model** | **Support Vector Machine (SVM)** | Finds the optimal separating hyperplane to classify SQLi vs. safe inputs. |
| **Statistical Model** | **Logistic Regression** | Predicts probability of input belonging to SQLi class using the sigmoid function. |
| **Deep Learning Model** | **Convolutional Neural Network (CNN)** | Extracts spatial patterns from text embeddings using convolutional and pooling layers; provides the highest accuracy. |

---

## 🧠 Methodology Overview

1. **Data Collection** – SQL Injection dataset (Kaggle) with 4200 labeled sentences  
   - Label `1`: SQL Injection  
   - Label `0`: Safe input  
2. **Data Preprocessing** –  
   - Removal of NULL values and duplicates  
   - Tokenization and text cleaning  
3. **Vectorization** –  
   - Using `CountVectorizer` (Bag-of-Words) to convert sentences into numerical features  
4. **Data Split** –  
   - 80% training, 20% testing for generalization  
5. **Model Training** –  
   - Train all six supervised models  
   - Hyperparameter tuning (epochs, batch size for CNN)  
6. **Evaluation** –  
   - Compare all models using `accuracy`, `precision`, `recall`, and `F1-score`

---

## 📊 Model Performance Summary

| Model | Type | Accuracy | Observation |
|--------|------|-----------|--------------|
| Naïve Bayes | Probabilistic | ≈ 0.95 | Fast and lightweight; excellent baseline |
| K-Nearest Neighbors | Instance-based | ≈ 0.90 | Sensitive to data scaling |
| Decision Tree | Tree-based | ≈ 0.91 | Interpretable but prone to overfitting |
| SVM | Margin-based | ≈ 0.93 | Strong generalization; effective on small data |
| Logistic Regression | Statistical | ≈ 0.92 | Good linear classifier for text data |
| **CNN** | Deep Learning | **≈ 0.97** | Best performer — captured complex injection patterns |

---

## 📈 Evaluation Metrics

The models were evaluated using:

- **Accuracy:** Percentage of correct predictions  
- **Precision:** Ratio of true positives to total predicted positives  
- **Recall:** Ratio of true positives to all actual positives  
- **F1-Score:** Harmonic mean of precision and recall  

**CNN achieved the highest F1-Score and precision**, showing strong robustness in differentiating SQLi strings from safe text.

---

## 🔍 Example Inference

| User Input | Model Prediction |
|-------------|------------------|
| `"my name is alex"` | ✅ Safe Input |
| `"1 or 1=1--"` | ⚠️ SQL Injection Detected |
| `"drop table users;"` | ⚠️ SQL Injection Detected |
| `"select * from accounts"` | ⚠️ SQL Injection Detected |

---

