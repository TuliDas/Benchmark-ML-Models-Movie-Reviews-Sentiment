
## 🧰 Project Structure
```
Benchmark-ML-Models-Movie-Reviews-Sentiment/
│
├── data/ 
|   └── dataset_link.md                    
│
├── notebooks/
│    ├── main.ipynb (#Colab)
│    └── IMDB_Movie_Reviews_Sentiment_Analysis.ipynb (with outputs)                  
│
├── src/  # All modular functions                  
|   ├── __init__.py
|   ├── data_loading.py
│   ├── data_preprocessing.py
│   ├── error_analysis.py
│   ├── feature_extraction.py
|   ├── hyperparameter_tuning.py
|   ├── model_evaluation.py
|   ├── model_performance_comparison.py
|   ├── model_training.py
|   ├── reporting.py
│   ├── select_best_models.py
│   └── utils.py
│ 
├── results/                  # All evaluation outputs
│   ├── metrics/              # Numeric/tabular results
│   │   ├── classification_metrics.md
│   │   ├── metrics_summary.json
│   │   ├── confusion_metrics_scores.md
│   │   ├── ConfisionMatrix-LinearSVC-tuned-BoW.png
│   │   └── ConfisionMatrix-LR-baseline-BoW.png
│   │
│   ├── reports/              # Human-readable analysis
│   │   ├── error_analysis.md
│   │   └── discussion.md
│   │
│   ├── fp_fn_lists/          # Plain text lists of misclassifications
│   │   ├── Both-Models' Predicted FP & FN list.txt
│   │   ├── Only-LogisticRegression-Baseline-(BoW)-FP & FN lists.txt
│   │   ├── Only-LinearSVC-Tuned-(TF-IDF)-FP & FN lists.txt
│   │   ├── fp-fn-of-best-two-models.png
│   │   └── fp-fn-overlaping.png
│   │
│   └── figures/              
│       └── baseline-vs-tuned-accuracy-comparison.png
│
└── README.md
```