# Benchmarking Supervised ML Models for Sentiment Classification (NLP) on IMDB Movie Reviews 🎬

This project focuses on building and evaluating machine learning models for sentiment classification of IMDB movie reviews (positive vs. negative). The workflow follows a structured and modular pipeline for preprocessing, feature extraction, model training, evaluation, selection, and error analysis.

---

## 📂 Dataset

- **Source:** [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Columns:** `review`, `sentiment`  
---

## 🧰 Project Structure
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full directory layout.
---

## 🔹 Workflow

1. **Data Loading & Preprocessing**  
   - Text cleaning, tokenization, stopword removal, stemming, and label binarization.  
   - Convert raw reviews into usable text format.

2. **Feature Extraction**  
   - Bag-of-Words (BoW)  
   - Term Frequency-Inverse Document Frequency (TF-IDF)

3. **Model Training, Hyperparameter Tuning & Evaluation**  
   - Models: Logistic Regression, MultinomialNB, SGDClassifier, LinearSVC  
   - Configurations: Baseline & Tuned  
   - [Metrics: Accuracy, Precision, Recall, F1-Score](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/metrics/classification_metrics.md)  

4. **Visualization**  
   - [Confusion matrices](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/metrics/confusion_metrics_scores.md)  
   - [Barplots for model performance comparisons](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/figures/baseline-vs-tuned-accuracy-comparison.png)

5. [**Best Model Selection**](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/best-two-models.png)  
   - Identify the **best (BoW) baseline** and **best (TF-IDF) tuned model** based on **F1-Score**.  
   - 💡 From the latest run of the notebook, the selected models were:  
     - **Logistic Regression** for baseline with **BoW** features  
     - **Linear SVC** for tuned model with **TF-IDF** features
     
6. **Error Analysis**  
   - Compute **False Positives (FP)** and **False Negatives (FN)** for both models.
   - [Generate detailed `.txt` files for FP & FN instances:](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/tree/main/results/fp_fn_lists)
     - Detected by both models
     - Detected by baseline only
     - Detected by tuned only  
   - [Compare misclassifications between baseline-BoW and tuned-Tf-Idf models.](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/reports/error_analysis.md)  
---
7. [**Discussion & Analysis of Results**](https://github.com/TuliDas/Benchmark-ML-Models-Movie-Reviews-Sentiment/blob/main/results/reports/discussion.md) 
   - **LinearSVC with TF-IDF (tuned)** gave the best results (~90% accuracy, F1).  
   - **Logistic Regression** was close, while **MultinomialNB** lagged behind.  
   - Models worked slightly better **without stemming**.  
   - Most errors came from **sarcasm, mixed sentiment, or long reviews**.  
   - Future improvements: try **n-grams** or **transformer-based models (e.g., BERT)** for better context handling. 
---

## How to Run

### Using Google Colab
1. Open the notebook: [`main.ipynb`](notebooks/main.ipynb)  
2. [Download the dataset]() , upload it to the colab   
3. The first cell contains a `git clone` command that will automatically download all the `src/` modules into the Colab environment. 
4. Run all cells **sequentially**.
5. Outputs, including model performance metrics, error analysis files, and visualizations, will be generated in Colab.  
---
