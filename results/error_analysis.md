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


#### **Example-09 (*Index: 9153*) (FN, Only LR-BoW)**

> "Of the films of the young republic few in number as they are The Buccaneer (1958) stands out as a finely crafted film... There are colorful side stories in this film of the young volunteer at his first dance to celebrate the victory."  

- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason :** The LR-BoW model misses the overall positive praise and focuses on neutral descriptive words like "battle," "march," "force," and "defend," which may seem negative or neutral in isolation. It fails to capture the celebratory tone and admiration for the film’s craftsmanship.  
- **Why SVC-TF-IDF got it right:** The SVC model with TF-IDF features accounts for term importance and context, giving more weight to sentiment-laden phrases like "finely crafted film," "celebrate the victory," and "colorful side stories." This allows it to correctly identify the overall positive sentiment despite the presence of neutral or seemingly negative descriptive words.  
- **Category:** Mixed sentiments / Long descriptive reviews


#### **Example-10 (*Index: 22885*) (FN, Only LR-BoW)**
> "Will all of you please lay the hell off Todd Sheets!?! Let's give you $30,000 to make a movie and see what you come up with! ... But what the hell, I still love this movie."  
- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason :** The LR-BoW model misinterprets negations and informal expressions like "not the worst either" and "hell" as negative, failing to capture the reviewer’s positive sentiment and overall praise for the movie and Todd Sheets’ effort.  
- **Why SVC-TF-IDF got it right:** The SVC model with TF-IDF accounts for the context and relative importance of words, recognizing key positive indicators such as "love this movie," "good old fashioned Guerilla Film-making," and "consummate professional," correctly identifying the overall positive sentiment despite the presence of negations and slang.  
- **Category:** Negations / Slang / Mixed sentiments


#### **Example-11 (*Index: 31041*) (FP, Only SVC-TF-IDF)**
> "The show had great episodes, this is not one of them. It's not a terrible episode, it's just hard to follow up 'The man that was death.', 'All through the house', and 'Dig that cat, he's real gone.'"  
- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason :** The SVC-TF-IDF model overemphasized the presence of positive words like "great" and "watch," while underestimating the negations and the subtle critique in "this is not one of them" and "hard to follow up." It thus misclassified the overall mildly negative review as positive.  
- **Why LR-BoW got it right:** The LR-BoW model, despite its simplicity, gives more weight to negation phrases and the relative absence of overwhelmingly positive sentiment across the review, correctly capturing the reviewer’s disappointment with this particular episode.  
- **Category:** Negations / Subtle critique / Mixed sentiments


#### **Example-12 (*Index: 17410*) (FP, Only SVC-TF-IDF)**
> "First of all, this plot is way overdone - girl wants to make it, everyone loves her, snobby girl intervenes, all looks lost, girl pulls through, everyone loves her again etc... I really hate how they keep on dissing classical music... the only reason that I can think of for making this movie is to promote Britney Spears... 1/10 stars."  

- **True Label:** Negative  
- **Predicted:** Positive  
- **Reason :** The SVC-TF-IDF model focused too much on positive or neutral words such as "good voice," "attractive," or "pulls through," and failed to properly account for strong negations and sarcastic phrases expressing dislike and criticism. It misclassified the overall negative review as positive.  
- **Why LR-BoW got it right:** The LR-BoW model captures negation and heavily repeated negative terms like "overdone," "hate," "insult," and the low rating "1/10," allowing it to correctly classify the review as negative.  
- **Category:** Negations / Sarcasm / Mixed sentiments


#### **Example-13 (*Index: 10896*) (FN, Only SVC-TF-IDF)**

> "I watch this movie at the start of every summer, and it never ceases to amuse me... Some of the jokes fall flat or will only elicit a slight chuckle, but others will leave you rolling... This is a good movie to watch over the summer... it's just funny as hell."  

- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason :** The SVC-TF-IDF model likely underestimated the overall positive sentiment because the review contains minor criticisms like "some jokes fall flat" and "slight chuckle." The model misinterpreted these as strong negative cues and failed to weigh the repeated positive phrases and superlatives like "never ceases to amuse," "leave you rolling," and "funny as hell."  
- **Why LR-BoW got it right:** The LR-BoW model emphasizes the frequent occurrence of clearly positive words such as "amuse," "funny," "hilarious," "good," and "bang," which outweigh the few minor criticisms. Its bag-of-words approach captures the dominant positive sentiment more reliably in this long, descriptive review.  
- **Category:** Mixed sentiments / Minor criticism within overall positive review


#### **Example-14 (*Index: 37966*) (FN, Only SVC-TF-IDF)**

> "Spoilers... I saw the original on TV sometime ago and remembered this production as less gripping than most Beeb costume drama... That said, I love these classic dramas and virtually all of them are a sight better than much of the 'modern' drama on TV these days. So 7 stars because in spite of the irritations it's still a good watch."  

- **True Label:** Positive  
- **Predicted:** Negative  
- **Reason :** The SVC-TF-IDF model overemphasized negative terms and phrases such as "less gripping," "irritated," "sanctimoniousness," "totally dissatisfied," and "ridiculous," causing it to misclassify the review despite the clear overall positive sentiment expressed at the end.  
- **Why LR-BoW got it right:** The LR-BoW model relies on the frequency of sentiment-bearing words and could pick up repeated positive terms like "good story," "commendations," "perfect," "handsome hero," and "love classic dramas," which outweighed the negative mentions and allowed it to correctly classify the review as positive.  
- **Category:** Mixed sentiments / Long descriptive reviews

---

## **4. Implications**
- **Mixed sentiment & contextual misunderstanding:** Many errors occur because the models cannot accurately capture shifts in sentiment or nuanced context within long reviews. Words expressing both positive and negative opinions in the same review often mislead the classifiers.
- **Sarcasm & subtle irony:** Reviews containing sarcasm or ironic phrasing are particularly challenging, as traditional bag-of-words or TF-IDF features fail to detect the intended sentiment.
- **Possible improvements**:
  - Preserve conjunctions like “while”, “but” to signal sentiment shifts.
  - Try n-grams (bi/tri) to capture short phrases instead of isolated words.
  - Explore transformer-based embeddings (BERT, DistilBERT) for context capture.
---
