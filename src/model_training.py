from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def train_logistic_regression(bow_reviews_train, tfidf_reviews_train ,sentiment_train, bow_reviews_test, 
                                                                            tfidf_reviews_test, sentiment_test):
    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    
    #Fitting the model for Bag of Words
    lr_bow = lr.fit(bow_reviews_train,sentiment_train)

    #Fitting the model for TF-IDF features
    lr_tfidf = lr.fit(tfidf_reviews_train,sentiment_train)
    

    #Predicting the model for bag of words & TF-IDF features
    lr_bow_pred = lr.predict(bow_reviews_test)
    lr_tfidf_pred = lr.predict(tfidf_reviews_test)

    return lr,lr_bow_pred, lr_tfidf_pred 



def train_multinomial_nb():
    return

def train_sgd():
    return

def train_svc():
    return
