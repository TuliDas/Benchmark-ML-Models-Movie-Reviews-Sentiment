from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def model_train(model):
    """
    Train a given model on both Bag of Words (BoW) and TF-IDF features, 
    then return predictions on the test set.

    Args:
        model: sklearn classifier instance (e.g., LogisticRegression, MultinomialNB, etc.)
    Returns:
        pred_for_bow (np.ndarray): Predictions on test set using BoW features.
        pred_for_tfidf (np.ndarray): Predictions on test set using TF-IDF features.
    """

    # Train separately
    model_bow = model.fit(bow_reviews_train, sentiment_train)
    model_tfidf = model.fit(tfidf_reviews_train, sentiment_train)

    # Predict on test data
    pred_for_bow = model_bow.predict(bow_reviews_test)
    pred_for_tfidf = model_tfidf.predict(tfidf_reviews_test)

    return pred_for_bow, pred_for_tfidf
