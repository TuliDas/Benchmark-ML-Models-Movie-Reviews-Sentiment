# Benchmarking Supervised ML Models for Sentiment Classification (NLP) on IMDB Movie Reviews 🎬

This project focuses on building and evaluating machine learning models for sentiment classification of IMDB movie reviews (positive vs. negative). The workflow follows a structured and modular pipeline for preprocessing, feature extraction, model training, evaluation, selection, and error analysis.

---

## 📂 Dataset

- **Source:** [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Columns:** `review`, `sentiment`  

---

## 🧰 Project Structure
```
Benchmarking-Supervised-ML-Models-for-Sentiment-Classification-NLP-on-IMDB-Reviews/
│
├── data/ # Raw and processed CSV datasets
├── src/ # All modular functions
| ├── __init__.py
| ├── data_loading.py
│ ├── data_preprocessing.py
│ ├── error_analysis.py
│ ├── feature_extraction.py
| ├── hyperparameter_tuning.py
| ├── model_evaluation.py
| ├── model_performance_comparison.py
| ├── model_training.py
| ├── reporting.py
│ ├── select_best_models.py
│ └── utils.py
├── notebooks/ #Colab notebooks - IMDB_Movie_Reviews_Sentiment_Analysis.ipynb
├── README.md
└── requirements.txt
```
---

## 🔹 Workflow

1. **Data Loading & Preprocessing**  
   - Text cleaning, tokenization, stopword removal, stemming, and label binarization.  
   - Convert raw reviews into usable text format.

2. **Feature Extraction**  
   - Bag-of-Words (BoW)  
   - Term Frequency-Inverse Document Frequency (TF-IDF)

3. **Model Training & Evaluation**  
   - Models: Logistic Regression, MultinomialNB, SGDClassifier, LinearSVC  
   - Configurations: Baseline & Tuned  
   - Metrics: Accuracy, Precision, Recall, F1-Score  

4. **Best Model Selection**  
   - Identify the **best baseline** and **best tuned model** based on **F1-Score**.

5. **Error Analysis**  
   - Compute **False Positives (FP)** and **False Negatives (FN)** for both models.  
   - Compare misclassifications between baseline and tuned models.  
   - Generate detailed `.txt` files for FP & FN instances:
     - Detected by both models
     - Detected by baseline only
     - Detected by tuned only  

6. **Visualization**  
   - Confusion matrices  
   - Barplots for model performance comparisons  

---


