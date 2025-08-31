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
├── notebooks/ #(Colab)-IMDB_Movie_Reviews_Sentiment_Analysis.ipynb
├── results/
| ├── Text files of all False-positives and false-negative/ #(.txt files of all False-Positives and False-Negatives)
| └── error_analysis.md 
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

3. **Model Training, Hypertuning & Evaluation**  
   - Models: Logistic Regression, MultinomialNB, SGDClassifier, LinearSVC  
   - Configurations: Baseline & Tuned  
   - Metrics: Accuracy, Precision, Recall, F1-Score  

4. **Visualization**  
   - Confusion matrices  
   - Barplots for model performance comparisons

5. **Best Model Selection**  
   - Identify the **best (BoW) baseline** and **best (tf-idf)tuned model** based on **F1-Score**.
     
6. **Error Analysis**  
   - Compute **False Positives (FP)** and **False Negatives (FN)** for both models.
   - [Generate detailed `.txt` files for FP & FN instances:](https://github.com/TuliDas/Benchmarking-Supervised-ML-Models-for-Sentiment-Classification-NLP-on-IMDB-Reviews/tree/main/results/Text%20files%20of%20all%20False-positives%20and%20false-negative)
     - Detected by both models
     - Detected by baseline only
     - Detected by tuned only  
   - [Compare misclassifications between baseline-BoW and tuned-Tf-Idf models.](https://github.com/TuliDas/Benchmarking-Supervised-ML-Models-for-Sentiment-Classification-NLP-on-IMDB-Reviews/blob/main/results/error_analysis.md)  

---

## How to Run

### Using Google Colab
1. Open the notebook: [`IMDB_Movie_Reviews_Sentiment_Analysis.ipynb`](notebooks/IMDB_Movie_Reviews_Sentiment_Analysis.ipynb)  
2. Run all cells **sequentially**.  
3. The first cell contains a `git clone` command that will automatically download all the `src/` modules into the Colab environment.  
4. Outputs, including model performance metrics, error analysis files, and visualizations, will be generated in Colab.  
---
