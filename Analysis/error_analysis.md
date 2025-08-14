# Error Analysis: LR vs SVC Models

The goal of this section is to understand where and why the models fail, and whether these errors follow any patterns.  
We compare two models trained on TF-IDF features:  

1. **LR Baseline – TF-IDF**  
2. **SVC Tuned – TF-IDF**  

We also analyze the impact of **stemming** on model performance.

---

## 1. Summary of Error Counts

Below is a comparison of false positives (FP) and false negatives (FN) for both models, **with and without stemming**.

### With Stemming

| Model                | False Positives | False Negatives | Accuracy | F1 Score |
|----------------------|----------------|----------------|----------|----------|
| LR Baseline TF-IDF   | 1199           | 953            | 0.8924   | 0.89238  |
| SVC Tuned TF-IDF     | 1093           | 940            | 0.89835  | 0.89834  |
| LR + SVC (Both) (FP) | 921            | ----           | ----     | ----     |
| LR + SVC (Both) (FN) | ----           | 733            | ----     | ----     |
| Only LR (not SVC)    | 278            | 220            | ----     | ----     |
| Only SVC (not LR)    | 172            | 207            | ----     | ----     |

### Without Stemming

| Model                | False Positives | False Negatives | Accuracy | F1 Score |
|----------------------|----------------|----------------|----------|----------|
| LR Baseline TF-IDF   | 1194           | 933            | 0.89365  | 0.89363  |
| SVC Tuned TF-IDF     | 1084           | 915            | 0.90005  | 0.90004  |
| LR + SVC (Both) (FP) | 908            | ----           | ----     | ----     |
| LR + SVC (Both) (FN) | ----           | 720            | ----     | ----     |
| Only LR (not SVC)    | 286            | 213            | ----     | ----     |
| Only SVC (not LR)    | 176            | 195            | ----     | ----     |

####**Notes**
- **SVC consistently better:** SVC has fewer FPs and FNs than LR, indicating it captures sentiment more accurately.  
- **Effect of stemming:** Removing stemming slightly improves accuracy and F1, suggesting over-normalization may remove meaningful distinctions.  
- **Shared vs unique errors:** Most errors are shared between models (inherently difficult reviews: mixed sentiment, negations, sarcasm, rare words), while unique errors point to model-specific weaknesses.  

---

## 2. Observations

- **SVC advantage:** Hyperparameter tuning improved SVC generalization, making it more robust than LR.  
- **Preprocessing artifacts:** Stemming can produce unnatural tokens (e.g., *"atroci"* from "atrocious"), slightly affecting interpretability.  
- **Complex reviews remain challenging:** Long sentences, contrast words, sarcasm, or mixed sentiment often lead to misclassification.  
- **Negation handling helps but has limits:** Joining negation with the next word (e.g., `never_good`) improves detection, but multiple clauses or long gaps still confuse models.


## **3. Examples**

> *(Excerpts shown for brevity — full reviews available in FP/FN index files)*

**Example (*Index: 9899*) (FP, SVC Only)**  
> "While an enjoyable movie to poke plot holes... ranks among the worst I've ever seen."  
- **True Label:** Negative  
- **Predicted:** Positive
- **Reason:** The contrast cue “while” is removed as a stopword, so early positive tokens (“enjoyable”) outweigh late negatives (“worst”, “mishmash”), tipping SVC to positive. 
- **Category:** Mixed Sentiment/Discourse cue removed


---

## **4. Implications**
- **Spelling-like tokens**: Lemmatization + stemming can produce tokens that look like spelling mistakes (e.g., “atroci”), but the issue isn’t human misspelling — it’s reduced semantic clarity for models that rely on context-less token counts.
- **Mixed sentiment & sarcasm**: Both models fail here because TF-IDF ignores sequence; contextual embeddings may help.
- **Possible improvements**:
  - Consider removing final stemming step; lemmatization may be sufficient.
  - Preserve conjunctions like “while”, “but” to signal sentiment shifts.
  - Try n-grams (bi/tri) to capture short phrases instead of isolated words.
  - Explore transformer-based embeddings (BERT, DistilBERT) for context capture.

---
