from sklearn.model_selection import GridSearchCV

# Define parameter grids for hyperparameter tuning
def get_hyperparameter_grids():
    """
    Define hyperparameter search spaces for different models.

    Returns:
        dict: Dictionary mapping model names to their hyperparameter grids.
    """
    param_grids = {}

    param_grids['LogisticRegression'] = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'max_iter': [100, 200, 500]
    }

    param_grids['MultinomialNB'] = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
    }

    param_grids['SGDClassifier'] = {
        'alpha': [1e-4, 1e-3, 1e-2],
        'loss': ['hinge', 'log'],   # hinge = linear SVM, log = logistic regression
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [1000, 2000]
    }

    param_grids['LinearSVC'] = {
        'C': [0.01, 0.1, 1, 10],
    }

    return param_grids

def tune_single_model(model, param_grid, X_train_vec, X_test_vec, y_train):
    """
    Perform hyperparameter tuning on a single model using GridSearchCV.

    Args:
        model: scikit-learn model instance
        param_grid (dict): Dictionary of hyperparameters to search
        X_train_vec (ndarray or sparse matrix): Training features (BoW or Tf-Idf)
        X_test_vec (ndarray or sparse matrix): Test features (BoW or Tf-Idf)
        y_train (array-like): Training labels

    Returns:
        tuple: (best_model, predictions, best_params)
            - best_model: The trained model with best hyperparameters
            - predictions: Predictions on test set
            - best_params: Best hyperparameters found
    """
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=2
    )

    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    predictions = best_model.predict(X_test_vec)

    return best_model, predictions, best_params

def tune_all_models(results, featured_data, param_grids, y_train, mode="tuned"):
    """
    Fine-tune all models with their parameter grids and update results dictionary.

    Args:
        results (dict): Nested results dictionary with model structure
        featured_data (dict): Dict with feature sets {'BoW': {...}, 'TF-IDF': {...}}
        param_grids (dict): Hyperparameter grids for all models
        y_train (array-like): Training labels
        mode (str): Either 'baseline' or 'tuned' (default = 'tuned')

    Returns:
        dict: Updated results dictionary with tuned predictions and best_params
    """
    for model_name, model_data in results.items():
        model = model_data["model"]
        param_grid = param_grids[model_name]

        for feat_type in featured_data.keys():
            X_train_vec = featured_data[feat_type]["train"]
            X_test_vec = featured_data[feat_type]["test"]

            best_model, pred, best_params = tune_single_model(
                model, param_grid, X_train_vec, X_test_vec, y_train
            )

            results[model_name][mode][feat_type]["predictions"] = pred
            results[model_name][mode][feat_type]["best_params"] = best_params
            results[model_name][mode][feat_type]["best_model"] = best_model

    return results

