**Note:** All scores shown in the table below are for data preprocessing **without stemming**

| #  | Model              | Feature | Version   | Accuracy | Precision | Recall  | F1-Score |
|----|--------------------|---------|-----------|----------|-----------|---------|----------|
| 0  | LogisticRegression | BoW     | baseline  | 0.89145  | 0.891519  | 0.89145 | 0.891445 |
| 1  | LogisticRegression | BoW     | tuned     | 0.89395  | 0.894022  | 0.89395 | 0.893945 |
| 2  | LogisticRegression | TF-IDF  | baseline  | 0.89355  | 0.893818  | 0.89355 | 0.893532 |
| 3  | LogisticRegression | TF-IDF  | tuned     | 0.89965  | 0.899808  | 0.89965 | 0.899640 |
| 4  | MultinomialNB      | BoW     | baseline  | 0.87355  | 0.873576  | 0.87355 | 0.873548 |
| 5  | MultinomialNB      | BoW     | tuned     | 0.87400  | 0.874015  | 0.87400 | 0.873999 |
| 6  | MultinomialNB      | TF-IDF  | baseline  | 0.87910  | 0.879234  | 0.87910 | 0.879089 |
| 7  | MultinomialNB      | TF-IDF  | tuned     | 0.88015  | 0.880310  | 0.88015 | 0.880137 |
| 8  | SGDClassifier      | BoW     | baseline  | 0.88450  | 0.884596  | 0.88450 | 0.884493 |
| 9  | SGDClassifier      | BoW     | tuned     | 0.88720  | 0.887790  | 0.88720 | 0.887157 |
| 10 | SGDClassifier      | TF-IDF  | baseline  | 0.89560  | 0.895947  | 0.89560 | 0.895577 |
| 11 | SGDClassifier      | TF-IDF  | tuned     | 0.89560  | 0.895947  | 0.89560 | 0.895577 |
| 12 | LinearSVC          | BoW     | baseline  | 0.88265  | 0.882682  | 0.88265 | 0.882648 |
| 13 | LinearSVC          | BoW     | tuned     | 0.89390  | 0.893998  | 0.89390 | 0.893893 |
| 14 | LinearSVC          | TF-IDF  | baseline  | 0.90000  | 0.900116  | 0.90000 | 0.899993 |
| 15 | LinearSVC          | TF-IDF  | tuned     | 0.90000  | 0.900116  | 0.90000 | 0.899993 |
