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

def evaluate_single_model(y_test, y_pred):
    """
    Evaluate a model's performance using accuracy, classification report, 
    confusion matrix, and a dictionary of detailed metrics.

    Args:
        y_test (array-like): True labels for the test set.
        y_pred (array-like): Predicted labels from the model.

    Returns:
        tuple:
            - acc_score (float): Accuracy score of the predictions.
            - report (str): Text summary of precision, recall, and f1-score for each class.
            - cm (ndarray): Confusion matrix as a 2D array.
            - report_dict (dict): Dictionary containing precision, recall, f1-score, and support 
              for each class and averaging method (e.g., 'macro avg', 'weighted avg').
    """
    acc_score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return acc_score, report, cm, report_dict


def evaluate_and_update_metrics(results, featured_data, y_test, mode="baseline"):
    """
    Evaluate predictions for each model and feature type in results
    and update accuracy, precision, recall, f1-score, classification report, 
    and confusion matrix.

    Args:
        results (dict): The main results dictionary storing models, predictions, and metrics.
        featured_data (dict): Dictionary containing BoW and TF-IDF train/test data.
        y_test (array-like): True labels for the test set.
        mode (str, optional): "baseline" or "tuned" to specify which predictions to evaluate.
                              Defaults to "baseline".

    Returns:
        dict: The updated results dictionary with evaluation metrics added for each model 
              and feature type ("BoW", "TF-IDF").
    """
    for model_name in results.keys():
        for feat_type in featured_data.keys():
            
            pred = results[model_name][mode][feat_type].get("predictions")
            acc, report, cm, report_dict = evaluate_single_model(y_test, pred)

            # Extract weighted average metrics
            precision = report_dict["weighted avg"]["precision"]
            recall = report_dict["weighted avg"]["recall"]
            f1_score = report_dict["weighted avg"]["f1-score"]

            # Update results dict
            results[model_name][mode][feat_type].update({
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "report": report,
                "confusion_matrix": cm
            })

    return results

