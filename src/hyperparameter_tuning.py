from sklearn.model_selection import GridSearchCV

def tuning_model(model, param_grid):
    """
    Perform hyperparameter tuning using GridSearchCV for both
    Bag of Words (BoW) and TF-IDF feature representations.

    Args:
        model : sklearn estimator
            The ML model (e.g., LogisticRegression, MultinomialNB, SGDClassifier, LinearSVC) 
            that will be tuned.
        
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter 
            settings to try as values.

    Returns:
        model_bow_pred_tuned : numpy.ndarray
            Predictions from the tuned model using Bag of Words features.
        
        model_tfidf_pred_tuned : numpy.ndarray
            Predictions from the tuned model using TF-IDF features.
    """

    # Run GridSearchCV with the given model and parameter grid.
    grid_model_bow = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_model_tfidf = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit the tuned model separately on BoW and TF-IDF training data.
    grid_model_bow.fit(bow_reviews_train, sentiment_train)
    grid_model_tfidf.fit(tfidf_reviews_train, sentiment_train)

    # Select the best estimator for each representation.
    # Predict sentiments on the test data.
    model_bow_pred_tuned = grid_model_bow.best_estimator_.predict(bow_reviews_test)
    model_tfidf_pred_tuned = grid_model_tfidf.best_estimator_.predict(tfidf_reviews_test)

    return model_bow_pred_tuned, model_tfidf_pred_tuned
