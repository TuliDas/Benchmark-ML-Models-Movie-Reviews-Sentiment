from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def initialize_model_dict_structure():
    """
    Initialize the dictionary structure to store machine learning models, their predictions, 
    evaluation metrics, and hyperparameter tuning results for both baseline and tuned models. 

    The structure supports two feature types: Bag-of-Words (BoW) and TF-IDF. For each model 
    and feature type, it provides placeholders for:
        - predictions
        - accuracy
        - cross-validation accuracy
        - precision, recall, f1-score
        - classification report
        - confusion matrix
        - best parameters and best model (for tuned models)
    """

    models = {
    "LogisticRegression": LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42),
    "MultinomialNB": MultinomialNB(),
    "SGDClassifier": SGDClassifier(random_state=42),
    "LinearSVC": LinearSVC(random_state=42)
      }

    # Initialize the structure
    results = {}

    for model_name, model in models.items():
      results[model_name] = {
        "model": model ,
        "baseline": {
            "BoW": {
                "predictions": None,
                "accuracy": None,
                "cv_accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "report": None ,
                "confusion_matrix": None
            },
            "TF-IDF": {
                "predictions": None,
                "accuracy": None,
                "cv_accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "report": None,
                "confusion_matrix": None
            }
        },
        "tuned": {
            "BoW": {
                "predictions": None,
                "accuracy": None,
                "cv_accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "report": None,
                "confusion_matrix": None,
                "best_params": None,
                "best_model": None
            },
            "TF-IDF": {
                "predictions": None,
                "accuracy": None,
                "cv_accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "report": None,
                "confusion_matrix": None,
                "best_params": None,
                "best_model": None
            }
        }
      }

    return results

def train_and_predict(model, featured_data_vec, y_train):
    """
    Train a scikit-learn model on a given feature set and return predictions 
    on the corresponding test data.

    Args:
        model (sklearn estimator): A scikit-learn model instance (e.g., LogisticRegression, MultinomialNB).
        featured_data_vec (dict): A dictionary containing vectorized features with keys:
                                  - 'train': training data (sparse matrix or array)
                                  - 'test' : test data (sparse matrix or array)
        y_train (array-like): True labels for the training data.

    Returns:
        ndarray: Predicted labels for the test set.
    """
    X_train_vec = featured_data_vec["train"]
    X_test_vec = featured_data_vec["test"]

    # Train the model
    model.fit(X_train_vec, y_train)

    # Make predictions on test data
    pred = model.predict(X_test_vec)

    return pred

def train_models_and_store_predictions(results, featured_data, y_train, mode="baseline"):
    """
    Train each model in results on all feature types in featured_data
    and store predictions in the structured results dictionary.

    Args:
        results (dict): Dictionary containing models and storage structure.
        featured_data (dict): Dictionary with keys as feature types
                              (e.g., 'BoW', 'TF-IDF') and values containing
                              'train' and 'test' data.
        y_train (array-like): Training labels.
        mode (str, optional): "baseline" or "tuned", specifying which section
                              of results to update. Defaults to "baseline".

    Returns:
        dict: Updated results dictionary with predictions stored under
              results[model_name][mode][feature_type]["predictions"].
    """
    for model_name in results.keys():
        model = results[model_name]["model"]

        for feat_type in featured_data.keys():   # loop dynamically over available feature types
            pred = train_and_predict(model, featured_data[feat_type], y_train)
            results[model_name][mode][feat_type]["predictions"] = pred

    return results

