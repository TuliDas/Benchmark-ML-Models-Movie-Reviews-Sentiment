import pandas as pd

def extract_metric_column(results, featured_data, metric_name):
    """
    Extract metric values (accuracy, precision, recall, f1_score) from results.

    Args:
        results (dict): Dictionary storing models, features, and evaluation results.
        featured_data (dict): Dictionary of feature representations (BoW, TF-IDF).
        metric_name (str): Name of the metric to extract (e.g. 'accuracy', 'precision').

    Returns:
        values: Flattened list of metric values for all models, features, and versions.
    """
    values = []
    for model_name, model_data in results.items():
        for feat_type in featured_data.keys():
            for version in ["baseline", "tuned"]:
                values.append(results[model_name][version][feat_type][metric_name])
    return values

def create_comparison_dataframe(results, featured_data):
    """
    Create a DataFrame comparing models across features, baseline/tuned versions, and metrics.

    Args:
        results (dict): Dictionary containing models and their evaluation results.
        featured_data (dict): Dictionary of feature representations (BoW, TF-IDF).

    Returns:
        pd.DataFrame: Comparison table with model, feature, version, and metrics.
    """
    model_names = []
    feature_types = []
    versions = []

    # Dynamically build labels
    for model_name in results.keys():
        for feat_type in featured_data.keys():
            for version in ["baseline", "tuned"]:
                model_names.append(model_name)
                feature_types.append(feat_type)
                versions.append(version)

    comparison_data_table = {
        "Model": model_names,
        "Feature": feature_types,
        "Version": versions,
        "Accuracy": extract_metric_column(results, featured_data, 'accuracy'),
        "Precision": extract_metric_column(results, featured_data, 'precision'),
        "Recall": extract_metric_column(results, featured_data, 'recall'),
        "F1-Score": extract_metric_column(results, featured_data, 'f1_score'),
    }
    df_results = pd.DataFrame(comparison_data_table)
    return df_results
