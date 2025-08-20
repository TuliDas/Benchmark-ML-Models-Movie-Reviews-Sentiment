import pandas as pd

def write_fp_fn(fp_df, fn_df, filename, title_fp, title_fn):
    """
    Save False Positives and False Negatives in a single text file with titles and counts.

    Parameters:
    fp_df      : pd.DataFrame of false positives
    fn_df      : pd.DataFrame of false negatives
    filename   : str, name of the output text file
    title_fp   : str, title for FP section
    title_fn   : str, title for FN section
    """
    with open(filename, "w", encoding="utf-8") as f:
        # Write FP section
        f.write(f"===== {title_fp} (Total: {len(fp_df)}) =====\n\n")
        for _, row in fp_df.iterrows():
            f.write(f"Index: {row['index']}\n")
            f.write(f"Original Review:\n{row['original_review']}\n\n")
            f.write(f"Processed Review:\n{row['processed_review']}\n\n")
            f.write(f"True Label: {row['true_label']}    Predicted: {row['predicted_label']}\n\n")
            f.write("---\n\n")

        # Write FN section
        f.write(f"\n===== {title_fn} (Total: {len(fn_df)}) =====\n\n")
        for _, row in fn_df.iterrows():
            f.write(f"Index: {row['index']}\n")
            f.write(f"Original Review:\n{row['original_review']}\n\n")
            f.write(f"Processed Review:\n{row['processed_review']}\n\n")
            f.write(f"True Label: {row['true_label']}    Predicted: {row['predicted_label']}\n\n")
            f.write("---\n\n")

    print(f"Saved '{filename}' with FP title '{title_fp}' and FN title '{title_fn}'.")

def create_three_text_files(separate_fp_fn_df):
    """
    Create three text files containing False Positives (FP) and False Negatives (FN) 
    for error analysis:
        1. Common FP/FN (both models)
        2. Unique FP/FN (only baseline)
        3. Unique FP/FN (only tuned)

    Args:
        separate_fp_fn_df (dict): Dictionary containing FP/FN DataFrames and filenames,
                                  produced by generate_error_analysis_separate_dataFrames().
    """

    # TextFile No-1 : FP & FN predicted by both models
    write_fp_fn(
        separate_fp_fn_df['both_models']['fp'],
        separate_fp_fn_df['both_models']['fn'],
        separate_fp_fn_df['both_models']['file-name'],
        title_fp="False Positives - Both Models",
        title_fn="False Negatives - Both Models"
    )

    # TextFile No-2 : FP & FN predicted only by baseline
    write_fp_fn(
        separate_fp_fn_df['only_baseline']['fp'],
        separate_fp_fn_df['only_baseline']['fn'],
        separate_fp_fn_df['only_baseline']['file-name'],
        title_fp="False Positives - Only Baseline",
        title_fn="False Negatives - Only Baseline"
    )

    # TextFile No-3 : FP & FN predicted only by tuned
    write_fp_fn(
        separate_fp_fn_df['only_tuned']['fp'],
        separate_fp_fn_df['only_tuned']['fn'],
        separate_fp_fn_df['only_tuned']['file-name'],
        title_fp="False Positives - Only Tuned",
        title_fn="False Negatives - Only Tuned"
    )
