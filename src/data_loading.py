
def load_imdb_data(nrows=None):
    """
    Args:
        csv_path (str): Path to the CSV file. If None, default path is used.
        nrows (int): Number of rows to load (for testing/debugging). If None, load all.
    
    Returns:
        data (pd.DataFrame): Loaded dataset (possibly truncated)
        data_OG (pd.DataFrame): Deepcopy of original dataset
    """
    
    csv_path = "https://raw.githubusercontent.com/TuliDas/IMDB-Movie-Reviews-Sentiment-Analysis/main/data/IMDB%20Dataset.csv"
    
    # Read CSV
    data = pd.read_csv(csv_path, engine='python', delimiter=',', nrows=nrows)
    
    # Keep an original copy for reference
    data_OG = copy.deepcopy(data)
    
    return data, data_OG

# Example usage when running directly
if __name__ == "__main__":
    data, data_OG = load_imdb_data(nrows=2000)  # only first 2000 rows for testing
    print("Data shape:", data.shape)
    print("First 5 rows:\n", data.head())

