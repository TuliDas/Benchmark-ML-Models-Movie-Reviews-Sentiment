# Error Analysis: LR vs SVC Models
The goal of this section is to understand where and why the models fail, and whether these errors follow any patterns.  
We compare two models trained on BoW and TF-IDF features:  

1. **LogisticRegrssion Baseline – BoW**  
2. **LinearSVC Tuned – TF-IDF**  

We also analyze the impact of **stemming** on model performance.

---

## 1. Summary of Error Counts

Below is a comparison of false positives (FP) and false negatives (FN) for both models, **with and without stemming**.

### With Stemming

| Model                | False Positives | False Negatives | Accuracy | F1 Score |
|----------------------|----------------|----------------|----------|----------|
| LR Baseline BoW      | 1161           | 1073           | 0.88830  | 0.888298 |
| SVC Tuned TF-IDF     | 1094           | 940            | 0.89830  | 0.898294 |
| LR + SVC (Both)      | 921            | 733            | ----     | ----     |
| Only LR (not SVC)    | 278            | 220            | ----     | ----     |
| Only SVC (not LR)    | 172            | 207            | ----     | ----     |

### Without Stemming

| Model                | False Positives | False Negatives | Accuracy | F1 Score |
|----------------------|----------------|----------------|----------|--------------------|
| LR Baseline BoW      | 1128           | 993            | 0.89145  | 0.891445199435332  |
| SVC Tuned TF-IDF     | 1085           | 915            | 0.9      | 0.8999927744779561 |
| LR + SVC (Both)      | 990            | 831            | ----     | ----               |
| Only LR (not SVC)    | 162            | 188            | ----     | ----               |
| Only SVC (not LR)    | 95             | 84             | ----     | ----               |


#### **Notes**
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

#### **Example-01 (Index: 2862) (FP, Both LR and SVC)**  
> "If you like me enjoy films with plots and convincing actors then ...........there must surely have been the spectre of lunacy in the room."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** Strong negative sentiment is present (“not the way to go,” “lunacy”), but humorous self-references, sarcasm, and playful exaggeration may have caused models to misinterpret the tone as lighthearted or neutral-positive.  
- **Category:** Sarcasm / Mixed Sentiment / Humor


#### **Example-02 (Index: 48505) (FP, Both LR and SVC)**  
> "I don't give much credence to AIDS conspiracy theories … and there is a sublime silent cameo by iconic performance artist, Ron Athey."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** Although the review emphasizes the film’s dark, ugly, and poorly done qualities, it also contains partially positive or neutral remarks (*“unique cultural context,” “partly saves,” “interesting performance,” “effective motif,” “sublime cameo”*). These scattered positive descriptors likely outweighed the overall negative tone, confusing the models into predicting positive. The mixed evaluation—criticism of the film with selective praise—makes the sentiment ambiguous.  
- **Category:** Mixed Sentiment / Contrast / Ambiguity  

#### **Example-03 (Index: 8200)(FP, Both LR and SVC)**  
> "If it wasn't for the terrific music, I would not hesitate to give this cinematic underachievement. But the music actually makes me like certain passages."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The positive phrases "terrific music" and "actually make like certain passage" dominated the sentiment in the processed text. Removing the numerical rating (2/10, 5/10) also removed a strong negative cue, causing both models to predict positive.  
- **Category:** Mixed Sentiment / Negation

#### **Example-04 (Index: 21078)FN, Both LR and SVC**  
> "A light-hearted, tongue-in-cheek family comedy that delivers laughs for both kids and adults, with Carey giving a great performance and festive cheer."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason:** The models likely focused on neutral or critic-related words like "comment," "reviewers," and comparisons to "Taxi Driver" or "The Godfather," failing to capture that the author was expressing a positive opinion despite criticizing others’ negativity.  
- **Category:** contextual misunderstanding ; positive sentiment missed due to neutral/critic words and 

 #### **Example-05 (Index: 26520)(FN, Both LR and SVC)**  
> "Just saw the movie, it's actually pretty good. Secret Agent OSS 117 is witty, with tongue-in-cheek humor, beautiful music, and a refreshing trip into the past with stylish settings."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason:** Positive aspects like humor, music, and nostalgia may have been overshadowed by words like "stupid" and "expensive," causing the model to misclassify.  
- **Category:** Mixed Sentiment / Humor / Irony

#### **Example-06 (Index: 20818)(FN, Both LR and SVC)**  
> "The main problem of the first 'Vampires' movie is that none of the characters were sympathetic. Carpenter learned from his mistake and this time used a likable vampire hunter and a charismatic vampire. The movie is generally enjoyable and ranks among the better entries to the genre."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason:** Despite mostly positive comments about likable characters and enjoyable aspects, the model likely focused on negative words like "problem" and "disappointed," leading to misclassification.  
- **Category:** Mixed Sentiment / Negation / Contrast

#### **Example-07 (*Index: 35467*) (FP, Only LR-BoW)** 
> ""Fly Me To The Moon" has to be the worst animated film I've seen in a LONG TIME. … The story is to be generous...trite. … Bad movie."  

- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The review is overwhelmingly negative, but LR-BoW likely overweighted isolated positive/neutral tokens such as *“OK 3-D visuals,”* *“mildly stimulating image,”* and *“pretty cool soundtrack.”* Since BoW ignores context and word order, these scattered positive words distorted the overall sentiment. The repeated strong negatives (*“worst,” “atrocious,” “bad movie”*) were diluted by the presence of a few positive terms.  
- **Why SVC got it right:** SVC with TF-IDF weighting downplays less informative or isolated positive words and focuses more on the strongly negative terms. The margin-based separation allowed SVC to correctly classify the dominant negative sentiment.  
- **Category:** Mixed Sentiment / Positivity bias from isolated tokens / Context loss in BoW  


#### **Example-08 (*Index: 22086*) (FP, Only LR-BoW)**
> "Free Willzyx … is the worst episode of ANY of the TV shows I watch… South Park was for very long my favorite… Free Willzyk has NONE of the content I mentioned earlier. It was so tame… I was extremely disgusted with this episode and I can't believe the shocking decline…"  

- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** LR-BoW appears to have been misled by strong positive associations with words like *“favorite,” “content,”* and *references to other popular shows* (Family Guy, Simpsons, etc.), while failing to give proper weight to intensifiers of negativity such as *“worst,” “tame,” “never get back,” “disgusted,”* and *“decline.”* The BoW model’s lack of context and order meant it overemphasized nostalgic or neutral references, diluting the clear negative sentiment.  

- **Why SVC got it right:** SVC with TF-IDF downweighted frequent neutral tokens (e.g., show titles, “content”) and gave stronger margin separation to the highly polar negative words like *“worst,” “tame,” “disgusted.”* Its margin-based learning allowed the end-of-review negative emphasis to dominate.  

- **Category:** Contrast / Contextual misunderstanding / Negation dilution  


#### **Example-09 (*Index: 30730*) (FN, Only LR)**   [UPDATE]
> "A decent little flick about a guy haunted by his past, with some quirky characters and a mix of madness .....A decent effort and worth seeing IF you like serial killer flicks."  
- **True Label:** Positive  
- **Predicted (LR):** Negative  
- **Reason:** Early negative terms (*"haunted"*, *"abusive"*, *"bonkers"*) describe the movie plot, not the reviewer’s sentiment. LR overemphasized these, underweighting later positive statements (*"decent effort"*, *"worth seeing"*).  
- **Why SVC got it right:** SVC balanced early negative and later positive terms better, giving higher weight to the evaluative parts that reflect the reviewer’s true opinion.  
- **Category:** Narrative-structure misinterpretation / Positive review with early negative plot description

#### **Example-10 (Index: 6778) (FN, Only LR)**   [UPDATE]
> Good movie…VERY good movie. … It really is the story that keeps you focused. .... more morbid horror fans and an interesting storyline."
- **True Label:** Positive  
- **Predicted (LR):** Negative
- **Reason:** The model likely misinterpreted mentions of "blood and gore," "vampires," and the director's usual reputation as negative, failing to account for the strong praise for story, acting, and overall enjoyment.
- **Category:** Contextual Misunderstanding / Focus on Surface Negatives

#### **Example-11 (*Index: 9899*) (FP, SVC Only)**   [UPDATE]
> "While an enjoyable movie to poke plot holes... ranks among the worst I've ever seen."  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason:** The contrast cue “while” was removed during stopword filtering, so early positive tokens (“enjoyable”) outweighed strong late negatives (“worst”, “mishmash”), tipping SVC toward a positive prediction. The long, clause-heavy structure made it harder for SVC’s TF-IDF features to capture the shift in sentiment.  
- **Why LR got it right:** Logistic Regression may have distributed weights more evenly, allowing multiple negative terms later in the review to collectively outweigh the single early positive cue. LR likely gave higher importance to terms like *"atrocious"*, *"worst"*, and *"confusing"*, preserving the overall negative classification despite the initial positivity.  
- **Category:** Mixed Sentiment / Discourse cue removed

#### **Example-12 (Index: 17210) (FP, Only SVC)**  [UPDATE]
> "Not only do alien visitors look exactly like furry armpitted human woman and not only are alien visitors able to perfectly speak English... This movie tries to get a moral, ecological point across but only succeeds in making you yawn and pray it ends soon."
- **True Label:** Negative
- **Predicted:** Positive
- **Reason:** The model likely misinterpreted descriptive words like "alien visitors" and "moral, ecological point" as neutral/positive content, while failing to capture the sarcastic and critical tone of the review that expresses boredom and annoyance.
- **Category:** Sarcasm / Surface Positives Misleading Overall Tone

#### **Example-13 (*Index: 2469*) (FN, Only SVC)**   [UPDATE]
> "It may be a remake of the 1937 film by Capra, but it is wrong to consider it only in that way! ... I strongly recommend it."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason:** The review contains strong positive phrases (*"excellent"*, *"terrific"*, *"strongly recommend"*) but also includes mild critical or contrastive statements early on. For SVC, TF-IDF likely assigned higher weight to negative terms like *"wrong"* and *"awful"*, overshadowing the overall positive tone. The long, multi-clause sentence structure diluted positive cues, making the model misinterpret the sentiment. 
- **Why LR got it right:** Logistic Regression may have balanced the influence of both positive and negative tokens more evenly, allowing the strong positive terms to dominate the final classification.  
- **Category:** Mixed cues with dominant positive sentiment

#### **Example-14 (Index: 11151) (FN, Only SVC)**     [UPDATE]
> "Wow! the French are really getting the hang of it... Mission Cleopatra is the best Asterix story ever written... not one moment you're bored... It's a must C!"
- **True Label:** Positive
- **Predicted:** Negative
- **Reason:** The review is highly positive, praising the story, acting, and humor. The model likely misclassified it because of mentions of minor criticisms (e.g., special effects, music choices) which created contrast words that the model interpreted as negative.
- **Category:** Mixed Sentiment / Minor Criticism Misread as Negative

---

## **4. Implications**
- **Mixed sentiment & contextual misunderstanding:** Many errors occur because the models cannot accurately capture shifts in sentiment or nuanced context within long reviews. Words expressing both positive and negative opinions in the same review often mislead the classifiers.
- **Sarcasm & subtle irony:** Reviews containing sarcasm or ironic phrasing are particularly challenging, as traditional bag-of-words or TF-IDF features fail to detect the intended sentiment.
- **Possible improvements**:
  - Preserve conjunctions like “while”, “but” to signal sentiment shifts.
  - Try n-grams (bi/tri) to capture short phrases instead of isolated words.
  - Explore transformer-based embeddings (BERT, DistilBERT) for context capture.
---
