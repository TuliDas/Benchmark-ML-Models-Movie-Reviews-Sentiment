# Discussion & Analysis of Results

This section synthesizes findings from the classification metrics and the detailed error analysis.  
_All scores were obtained using preprocessing **without stemming**._

---
## Overall Performance
- All models achieved strong results (>87% accuracy), which indicates that the dataset is suitable for binary sentiment classification.
- Performance differences between models are relatively small, but consistent trends can be observed.

## Performance Comparison Between Models
- **LinearSVC (TF-IDF, tuned)** achieved the best overall performance:
  - Accuracy ≈ **0.900**
  - F1-Score ≈ **0.899**
- **Logistic Regression** improved notably after tuning (0.891 → 0.899).Balanced performance, interpretable, and efficient for mid-scale applications.
- **MultinomialNB** Fastest and lightweight, but performed the weakest (≈0.874–0.880) Suitable for resource-limited setups.
- **SGDClassifier** showed balanced results, close to SVC in TF-IDF settings.
---

## Feature Representation
- **TF-IDF consistently outperformed BoW** across all models.
- Reason: TF-IDF emphasizes important terms rather than raw frequency, improving the ability to distinguish subtle sentiment cues.
---

## Effect of Hyperparameter Tuning
- **Most beneficial for Logistic Regression**, with ~0.8% accuracy boost.
- **SVC and SGD** improved slightly, confirming they are already robust in baseline form.
- **MultinomialNB** had almost no gain (0.873 → 0.880), suggesting its simplicity limits sensitivity to hyperparameter adjustments.
---

## Error Analysis Insights (Summary)
- **SVC consistently makes fewer errors than LR** (both FP and FN).
- **Stemming slightly worsened performance**, likely due to over-normalization (e.g., *atroci* from “atrocious”).
- **Shared errors** across models are often due to:
  - Sarcasm and irony.
  - Mixed or shifting sentiments in long reviews.
  - Negations spread across clauses.
- **Unique errors** highlight model-specific weaknesses (e.g., LR being more sensitive to vocabulary overlap).

---

## Future Improvements
- Preprocessing:  
  - Preserve conjunctions like *but, while* to capture sentiment shifts.  
  - Use n-grams (bi/tri) to capture short phrases instead of isolated words (e.g., *not good*).  
- Modeling:  
  - Explore **transformer-based embeddings (BERT, DistilBERT)** for contextual understanding.  
  - Try **ensembling models** to reduce unique error sets.  
- Error-specific refinements:  
  - Better negation handling.  
  - Sarcasm detection mechanisms.  

---

## File Organization
- **`report/error_analysis.md`** → Detailed per-example error deep dive.  
- **`report/discussion.md`** → High-level summary, trends, and future work.  

---
