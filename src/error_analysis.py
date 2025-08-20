import pandas as pd
import numpy as np

def initialize_error_analysis_dict(best_two_models):

  error_analysis_dict = {} 

  for version, info in best_two_models.items():
    
    error_analysis_dict[version] = {
        "full-name" : info["full-name"],
        "name" : info["name"],
        "feature" : info["feature"],
        "f1_score" : info["f1-score"],
        "false_positives": None,
        "false_negatives": None,
        "fp_indexes": None,
        "fn_indexes": None
    }
  return error_analysis_dict

def calculate_singleModel_fp_fn(y_pred, data_OG, data, X_test, y_test):
    """
    Returns False Positives and False Negatives DataFrames along with counts and index lists.

    Parameters:
    y_pred         : array-like, model's predicted labels
    data_OG        : pd.DataFrame, original reviews
    data           : pd.DataFrame, processed reviews
    X_test         : pd.Series, test reviews (to get indices)
    y_test         : array-like, true labels

    Returns:
    false_positives : pd.DataFrame with index, original review, processed review, true & predicted labels
    false_negatives : pd.DataFrame with index, original review, processed review, true & predicted labels
    fp_indexes      : list, indexes of false positives
    fn_indexes      : list, indexes of false negatives
    """
    test_idx = X_test.index

    # Build error analysis DataFrame
    error_df = pd.DataFrame({
        'index': test_idx,
        'original_review': data_OG.loc[test_idx, 'review'].values,
        'processed_review': data.loc[test_idx, 'review'].values,
        'true_label': np.array(y_test).ravel(),
        'predicted_label': np.array(y_pred).ravel()
    })

    # Identify false positives and false negatives
    false_positives = error_df[(error_df['predicted_label'] == 1) & (error_df['true_label'] == 0)]
    false_negatives = error_df[(error_df['predicted_label'] == 0) & (error_df['true_label'] == 1)]


    # Index lists
    fp_indexes = false_positives['index'].tolist()
    fn_indexes = false_negatives['index'].tolist()

    return false_positives, false_negatives, fp_indexes, fn_indexes

def compute_bestModels_all_fp_fn(error_analysis_dict, best_two_models, data_OG, data, X_test, y_test):
  """
  Args:
        error_analysis_dict (dict): Dictionary to update with FP/FN details.
        best_two_models (dict): Dictionary returned by get_best_baseline_and_tuned(), 
                                containing predictions for 'baseline' and 'tuned'.
        data_OG (pd.DataFrame): Original dataset (with raw reviews).
        data (pd.DataFrame): Preprocessed dataset (with processed reviews).
        X_test (pd.Series): Test feature input (to align indices).
        y_test (array-like): True labels for test set.

    Returns:
        dict: Updated error_analysis_dict with FP/FN results for both models.
    """
  for version, info in best_two_models.items():
    predictions = best_two_models[version]["predictions"]
    fp , fn , fp_indexes , fn_indexes = calculate_singleModel_fp_fn(predictions, data_OG, data, X_test, y_test)
    error_analysis_dict[version].update ({
        "false_positives": fp,
        "false_negatives": fn,
        "fp_indexes": fp_indexes,
        "fn_indexes": fn_indexes
    })
  return error_analysis_dict


def generate_error_analysis_separate_dataFrames(error_analysis_dict, baseline_name, tuned_name):
    """
    Generate DataFrames of False Positives (FP) and False Negatives (FN) 
    for both baseline and tuned models, separating common and unique errors.

    Args:
        error_analysis_dict (dict): Dictionary storing FP, FN and their indexes for models.
        baseline_name (str): Full descriptive name of the baseline model.
        tuned_name (str): Full descriptive name of the tuned model.

    Returns:
        dict: Dictionary with 3 entries:
            - 'both_models': FP/FN common to both models
            - 'only_baseline': FP/FN unique to baseline
            - 'only_tuned': FP/FN unique to tuned

        Each entry is itself a dict with:
            - "file-name": Suggested filename for saving results
            - "fp": DataFrame of false positives
            - "fn": DataFrame of false negatives
    """

    FP_set_baseline = set(error_analysis_dict['baseline']['fp_indexes'] )
    FN_set_baseline = set(error_analysis_dict['baseline']['fn_indexes'])
    FP_set_tuned = set(error_analysis_dict['tuned']['fp_indexes'] )
    FN_set_tuned = set(error_analysis_dict['tuned']['fn_indexes'])

    baseline_fp_df = error_analysis_dict['baseline']['false_positives']
    baseline_fn_df = error_analysis_dict['baseline']['false_negatives']
    tuned_fp_df = error_analysis_dict['tuned']['false_positives']
    tuned_fn_df = error_analysis_dict['tuned']['false_negatives']

    separate_fp_fn_df = {} 
    # FP and FN presents in both baseline and tuned
    FP_set_bothModels = FP_set_baseline & FP_set_tuned
    FN_set_bothModels = FN_set_baseline & FN_set_tuned
    FP_df_bothModels = baseline_fp_df[baseline_fp_df['index'].isin(FP_set_bothModels)]   # can also be calculated using tuned_fp_df (works vice-versa)
    FN_df_bothModels = baseline_fn_df[baseline_fn_df['index'].isin(FN_set_bothModels)]
    
    separate_fp_fn_df['both_models'] = {
        "file-name" : f"Both-Models' Predicted FP & FN list.txt",
        "fp" : FP_df_bothModels,
        "fn" : FN_df_bothModels
    }

    # FP and FN that only in baseline
    FP_set_onlyBaseline = FP_set_baseline - FP_set_tuned
    FN_set_onlyBaseline = FN_set_baseline - FN_set_tuned
    FP_df_onlyBaseline = baseline_fp_df[baseline_fp_df['index'].isin(FP_set_onlyBaseline)]
    FN_df_onlyBaseline = baseline_fn_df[baseline_fn_df['index'].isin(FN_set_onlyBaseline)]
    
    separate_fp_fn_df['only_baseline'] = {
        "file-name" : f"Only-{baseline_name}-FP & FN lists.txt",
        "fp" : FP_df_onlyBaseline,
        "fn" : FN_df_onlyBaseline
    }
   

    # FP and FN that only in tuned
    FP_set_onlyTuned = FP_set_tuned - FP_set_baseline
    FN_set_onlyTuned = FN_set_tuned - FN_set_baseline
    FP_df_onlyTuned = tuned_fp_df[tuned_fp_df['index'].isin(FP_set_onlyTuned)]
    FN_df_onlyTuned = tuned_fn_df[tuned_fn_df['index'].isin(FN_set_onlyTuned)]
    separate_fp_fn_df['only_tuned'] = {
        "file-name" : f"Only-{tuned_name}-FP & FN lists.txt",
        "fp" : FP_df_onlyTuned,
        "fn" : FN_df_onlyTuned
    } 

    return separate_fp_fn_df     # return dict containing the 6 DataFrames

