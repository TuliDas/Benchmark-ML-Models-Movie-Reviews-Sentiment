**Note:** All scores shown in the table below are for data preprocessing **without stemming**

| #  | Model              | Feature | Version   | True-Positive | True-Negative | False-Positive  | False-Negative |
|----|--------------------|---------|-----------|---------------|---------------|-----------------|----------------|
| 0  | LogisticRegression | BoW     | baseline  | 8981          | 8848          | 1152            | 1019           |
| 1  | LogisticRegression | BoW     | tuned     | 9007          | 8872          | 1128            | 993            |
| 2  | LogisticRegression | TF-IDF  | baseline  | 9066          | 8805          | 1195            | 934            |
| 3  | LogisticRegression | TF-IDF  | tuned     | 9096          | 8897          | 1103            | 904            |
| 4  | MultinomialNB      | BoW     | baseline  | 8777          | 8694          | 1306            | 1223           |
| 5  | MultinomialNB      | BoW     | tuned     | 8772          | 8708          | 1292            | 1228           |
| 6  | MultinomialNB      | TF-IDF  | baseline  | 8885          | 8697          | 1303            | 1115           |
| 7  | MultinomialNB      | TF-IDF  | tuned     | 8904          | 8699          | 1301            | 1096           |
| 8  | SGDClassifier      | BoW     | baseline  | 8924          | 8766          | 1234            | 1076           |
| 9  | SGDClassifier      | BoW     | tuned     | 9067          | 8677          | 1323            | 933            |
| 10 | SGDClassifier      | TF-IDF  | baseline  | 9104          | 8808          | 1192            | 896            |
| 11 | SGDClassifier      | TF-IDF  | tuned     | 9104          | 8808          | 1192            | 896            |
| 12 | LinearSVC          | BoW     | baseline  | 8872          | 8781          | 1219            | 1128           |
| 13 | LinearSVC          | BoW     | tuned     | 9018          | 8860          | 1140            | 982            |
| 14 | LinearSVC          | TF-IDF  | baseline  | 9085          | 8915          | 1085            | 915            |
| 15 | LinearSVC          | TF-IDF  | tuned     | 9085          | 8915          | 1085            | 915            |
