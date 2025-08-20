import pandas as pd

def get_best_baseline_and_tuned(df_results, results, metric="F1-Score"):
    """
    Retrieve the best baseline and tuned models based on a specified evaluation metric.

    Args:
        df_results (pd.DataFrame): DataFrame containing model comparison metrics,
                                   with columns including "Model", "Feature", "Version", and metric columns.
        results (dict): Nested dictionary storing model objects and predictions.
        metric (str, optional): The evaluation metric to rank models by 
                                (e.g., "F1-Score", "Accuracy"). Default = "F1-Score".

    Returns:
        dict: Dictionary with keys {"baseline", "tuned"} where each entry contains:
              - full_name (str): Descriptive model name (Model-Version-(Feature))
              - model (str): Model name
              - feature (str): Feature representation (e.g., TF-IDF, BoW)
              - predictions (np.ndarray): Model predictions
              - metric (float): Value of the selected evaluation metric
    """
    best_two_models = {}

    # Best baseline
    baseline_df = df_results[df_results["Version"].str.lower() == "baseline"]
    best_baseline = baseline_df.sort_values(by=metric, ascending=False).iloc[0]

    full_name = f"{best_baseline['Model']}-Baseline-({best_baseline['Feature']})"
    best_two_models["baseline"] = {
        "full-name" : full_name,
        "name": best_baseline['Model'],
        "feature": best_baseline['Feature'],
        "predictions": results[best_baseline['Model']]["baseline"][best_baseline['Feature']]["predictions"],
        "f1-score": best_baseline[metric] ,
    }

    # Best tuned
    tuned_df = df_results[df_results["Version"].str.lower() == "tuned"]
    best_tuned = tuned_df.sort_values(by=metric, ascending=False).iloc[0]

    full_name = f"{best_tuned['Model']}-Tuned-({best_tuned['Feature']})"
    best_two_models["tuned"] = {
        "full-name" : full_name,
        "name" : best_tuned['Model'],
        "feature" : best_tuned['Feature'],
        "predictions": results[best_tuned['Model']]["tuned"][best_tuned['Feature']]["predictions"],
        "f1-score" : best_tuned[metric]
    }

    return best_two_models
