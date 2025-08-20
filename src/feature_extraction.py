from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def initialize_featured_data():
    """
    Initialize dictionary to hold feature extraction methods and their transformed data.

    Returns:
        dict: {
            "BoW": {"vectorizer": CountVectorizer, "train": None, "test": None},
            "TF-IDF": {"vectorizer": TfidfVectorizer, "train": None, "test": None}
        }
    """
    feature_type = {
        "BoW": CountVectorizer(min_df=5, max_df=0.9, binary=False, ngram_range=(1, 3)),
        "TF-IDF": TfidfVectorizer(min_df=5, max_df=0.9, use_idf=True, ngram_range=(1, 3))
    }

    featured_data = {}
    for name, vec in feature_type.items():
        featured_data[name] = {
            "vectorizer": vec,
            "train": None,
            "test": None
        }
    return featured_data

def extract_single_feature(vectorizer, X_train, X_test):
    """
    Transform raw text data into numerical feature representations.

    Args:
        vectorizer (sklearn vectorizer): e.g. CountVectorizer or TfidfVectorizer
        X_train (list or array-like): Training text data
        X_test (list or array-like): Testing text data

    Returns:
        dict: {
            "vectorizer": fitted vectorizer,
            "train": transformed X_train,
            "test": transformed X_test
        }
    """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return {"vectorizer": vectorizer, "train": X_train_vec, "test": X_test_vec}

def extract_all_features(featured_data, X_train, X_test):
    """
    Fit and transform all initialized vectorizers on training and test data.

    Args:
        featured_data (dict): Dictionary from initialize_featured_data()
        X_train (list or array-like): Training text data
        X_test (list or array-like): Testing text data

    Returns:
        dict: Updated featured_data with fitted vectorizers and transformed train/test sets
    """
    for name, details in featured_data.items():
        vec = details["vectorizer"]
        featured_data[name] = extract_single_feature(vec, X_train, X_test)
    return featured_data


