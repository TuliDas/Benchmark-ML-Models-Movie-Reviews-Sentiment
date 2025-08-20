# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, matrix_title, class_labels=['Negative', 'Positive']):
    """
    Visualize the confusion matrix using a heatmap.

    Args:
        cm (array-like): Confusion matrix values (2D array).
        matrix_title (str): Title to display on the heatmap.
        class_labels (list, optional): List of class names for the axes. 
            Default is ['Negative', 'Positive'].

    Returns:
        None: Displays the heatmap plot.
    """
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(matrix_title)
    plt.show()

def visualize_confusion_matrix(results,featured_data, mode="baseline"):

  for model_name in results.keys():

      for feat_type in featured_data.keys():
          cm = results[model_name][mode][feat_type]["confusion_matrix"]
          matrix_title = f"{model_name}-{mode} ( {feat_type} ) Confusion Matrix"

          plot_confusion_matrix(cm, matrix_title)

def visualize_model_comparison(df_results):
  plt.figure(figsize=(12,6))
  sns.barplot(data=df_results, x="Model", y="Accuracy", hue="Version")
  plt.title("Model Accuracy Comparison: Baseline vs Tuned")
  plt.show()
