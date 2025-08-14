# IMDB-Movie-Reviews-Sentiment-Analysis
# Error Analysis

The goal of this section is to understand where and why the models fail, and whether these errors follow any patterns.  
We compare two models:
1. **LR Baseline – TF-IDF**
2. **SVC Tuned – TF-IDF**

For each, we analyze:
- False Positives (FP): Negative reviews predicted as Positive
- False Negatives (FN): Positive reviews predicted as Negative
### **Total Reviews = 50,000**
| Model               | False Positives | False Negatives |
|---------------------|----------------|----------------|
| LR Baseline TF-IDF  | 1199           | 953             |
| SVC Tuned TF-IDF    | 1093           | 940             |
| LR + SVC(Both) (FP) | 921               |  ----           |
| LR + SVC(Both) (FN) |---             |  733              |
| Only LR (not SVC)    |   278             |   220              |
| Only SVC (not LR)    |   172             |  207               |


Total False Positives(FP) in LR_Baseline_TFIDF : 1199
Total False Negatives(FN) in LR_Baseline_TFIDF : 953
Total False Positives(FP) in SVC_Tuned_TFIDF : 1093
Total False Negatives(FN) in SVC_Tuned_TFIDF : 940

Total Overlap(FP) in both LR & SVC : 921
Total Overlap(FN) in both LR & SVC : 733

FP only in LR : 278
FN only in LR : 220
FP only in SVC : 172
FN only in SVC : 207
