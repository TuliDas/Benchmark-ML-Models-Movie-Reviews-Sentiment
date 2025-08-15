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
### **Example (*Index: 23345*) (FP, Both LR and SVC)**  
> "What a truly moronic movie, all I can say is the writer must be very fond of magic mushrooms and LSD because this must be the result of one of his 'trips'..."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The review mixes strong negative language (*"moronic"*, *"disappoint"*) with humorous, sarcastic storytelling and neutral descriptive content about confusing plot points. Both models misclassified it because TF-IDF emphasizes frequent neutral/positive tokens over subtle negative/sarcastic expressions.  
- **Category:** Mixed Sentiment / Sarcasm

### **Example (Index: 31689) (FP, Both LR and SVC)**  
> "I didn't hate this movie as much as some on my all-time blacklist ..... Scene one is very good, all the rest are crap."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The early use of “good” for the first scene may outweigh later negative descriptors like “waste” and “crap.” The humorous references to actors could further confuse sentiment, causing both models to predict positive.  
- **Category:** Mixed Sentiment / Slang / Humor
 

**Example (*Index: 5464*) (FP, Only LR)**  
> "Begins better than it ends... The message deciphered was contrary to the whole story. It just does not mesh."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** Early positive phrasing (“begins better”) and light humor (“funny”) may have been overweighted by LR’s TF-IDF coefficients, overshadowing the later clear negative statements (“does not mesh”, “contrary to the whole story”). LR’s linear nature makes it more prone to being swayed by strong early positive terms if the later negatives have slightly weaker learned weights.  
- **Why SVC got it right:** SVC’s margin-based classification might have better separated the negative terms in vector space, allowing the stronger negative conclusion to dominate despite the initial positive cues.  
- **Category:** Mixed Sentiment / Positivity bias from early clauses

**Example (*Index: 30730*) (FN, Only LR)**  
> "A decent little flick about a guy haunted by his past, with some quirky characters and a mix of madness .....A decent effort and worth seeing IF you like serial killer flicks."  

- **True Label:** Positive  
- **Predicted (LR):** Negative  
- **Reason:** Early negative terms (*"haunted"*, *"abusive"*, *"bonkers"*) describe the movie plot, not the reviewer’s sentiment. LR overemphasized these, underweighting later positive statements (*"decent effort"*, *"worth seeing"*).  
- **Why SVC got it right:** SVC balanced early negative and later positive terms better, giving higher weight to the evaluative parts that reflect the reviewer’s true opinion.  
- **Category:** Narrative-structure misinterpretation / Positive review with early negative plot description


**Example (*Index: 9899*) (FP, SVC Only)**  
> "While an enjoyable movie to poke plot holes... ranks among the worst I've ever seen."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The contrast cue “while” was removed during stopword filtering, so early positive tokens (“enjoyable”) outweighed strong late negatives (“worst”, “mishmash”), tipping SVC toward a positive prediction. The long, clause-heavy structure made it harder for SVC’s TF-IDF features to capture the shift in sentiment.  
- **Why LR got it right:** Logistic Regression may have distributed weights more evenly, allowing multiple negative terms later in the review to collectively outweigh the single early positive cue. LR likely gave higher importance to terms like *"atrocious"*, *"worst"*, and *"confusing"*, preserving the overall negative classification despite the initial positivity.  
- **Category:** Mixed Sentiment / Discourse cue removed


**Example (*Index: 2469*) (FN, Only SVC)**  
> "It may be a remake of the 1937 film by Capra, but it is wrong to consider it only in that way! ... I strongly recommend it."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason:** The review contains strong positive phrases (*"excellent"*, *"terrific"*, *"strongly recommend"*) but also includes mild critical or contrastive statements early on. For SVC, TF-IDF likely assigned higher weight to negative terms like *"wrong"* and *"awful"*, overshadowing the overall positive tone. The long, multi-clause sentence structure diluted positive cues, making the model misinterpret the sentiment. 
- **Why LR got it right:** Logistic Regression may have balanced the influence of both positive and negative tokens more evenly, allowing the strong positive terms to dominate the final classification.  
- **Category:** Mixed cues with dominant positive sentiment


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
