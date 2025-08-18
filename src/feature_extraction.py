from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def extract_features(vectorizer, X_train, X_test, name):
    """
    Transform raw text data into numerical feature representations.
    
    Args:
        vectorizer: sklearn vectorizer (e.g. CountVectorizer, TfidfVectorizer)
        X_train (list or array): training text data
        X_test (list or array): testing text data
        name (str): key name to store results (e.g. "BoW", "TFIDF")
    
    Returns:
        dict: {name: {"train": X_train_vec, "test": X_test_vec}}
    """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return {
        name: { "train": X_train_vec, "test": X_test_vec   }
           }
