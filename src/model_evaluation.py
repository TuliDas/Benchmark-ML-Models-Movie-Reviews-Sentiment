from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def calculate_cross_validation(results, featured_data, y_train, mode="baseline"):
    """
    Calculate 5-fold cross-validation accuracy for each model and feature type
    and store it in the results dictionary.

    Args:
        results (dict): Dictionary storing models and predictions.
        featured_data (dict): Dictionary containing BoW and TF-IDF train/test data.
        y_train (array-like): Training labels.
        mode (str): 'baseline' or 'tuned' to specify which results to update.

    Returns:
        dict: Updated results dictionary with cross-validation accuracy added.
    """
    for model_name, model_info in results.items():
        model = model_info["model"]

        for feat_type in featured_data.keys():
            X_train_vec = featured_data[feat_type]["train"]
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="accuracy")
            results[model_name][mode][feat_type]["cv_accuracy"] = cv_scores.mean()

    return results
