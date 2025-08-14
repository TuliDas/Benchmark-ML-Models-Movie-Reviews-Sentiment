# Error Analysis

The goal of this section is to understand where and why the models fail, and whether these errors follow any patterns.  
We compare two models:
1. **LR Baseline – TF-IDF**
2. **SVC Tuned – TF-IDF**

For each, we analyze:
- **False Positives (FP):** Negative reviews predicted as Positive  
- **False Negatives (FN):** Positive reviews predicted as Negative  

---

## **1. Summary of Error Counts**

| Model               | False Positives | False Negatives |
|---------------------|----------------|----------------|
| LR Baseline TF-IDF  | 1199           | 953            |
| SVC Tuned TF-IDF    | 1093           | 940            |
| LR + SVC (Both) (FP)| 921            | ----           |
| LR + SVC (Both) (FN)| ----           | 733            |
| Only LR (not SVC)   | 278            | 220            |
| Only SVC (not LR)   | 172            | 207            |

---

## **2. Observations**
- **Large error overlap**: Many reviews (921 FP, 733 FN) are misclassified by *both* models — these are likely inherently ambiguous or lose crucial sentiment clues after preprocessing.
- **SVC advantage**: SVC has fewer unique errors compared to LR, suggesting the hyperparameter tuning improved generalization.
- **Preprocessing artifacts**: Stemming sometimes produces unnatural tokens (e.g., *"atroci"* from "atrocious") — while not “spelling mistakes” in the traditional sense, these altered forms may reduce model interpretability.
- **Negation handling helped**: Negation words were preserved using token joining (e.g., `never_good`), but cases with long gaps or multiple clauses still confuse both models.
- **Mixed sentiment still tricky**: Reviews that start positive and end negative (or vice versa) are often misclassified due to TF-IDF’s lack of word order awareness.

---

## **3. Examples**

> *(Excerpts shown for brevity — full reviews available in FP/FN index files)*

**Example (*Index: 9899*) (FP, SVC Only)**  
> "While an enjoyable movie to poke plot holes... ranks among the worst I've ever seen."  
- **True Label:** Negative  
- **Predicted:** Positive
- **LR Predicted:** Negative (Correct)
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
