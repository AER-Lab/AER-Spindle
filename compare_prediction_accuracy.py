import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(data, labels):
    # Assuming `data` is the predictions and `labels` are the actual labels
    return pd.crosstab(labels, data, normalize='index') * 100

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_class_accuracy(class_accuracy, title):
    # Create the bar plot
    ax = class_accuracy.plot(kind='bar', color='skyblue', figsize=(10, 10))
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # Annotate the bars with the percentage values
    for p in ax.patches:  # Loop over every bar
        ax.annotate(f"{p.get_height():.2f}%",  # This formats the number as a percentage
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # This positions the text in the center of the bar
                    ha='center',  # Horizontally center the text
                    va='center',  # Vertically center the text
                    xytext=(0, 10),  # Position text 10 points above the top of the bar
                    textcoords='offset points')  # Use offset points to position the text

    plt.show()

def compare_predictions_and_labels(prediction_file, label_file, writer, sheet_name):
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    prediction_file_name = os.path.basename(prediction_file)
    label_file_name = os.path.basename(label_file)
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    print("Comparing {} and {}".format(prediction_file_name, label_file_name))
    
    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['Prediction'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Calculate overall accuracy
    overall_accuracy = combined['Correct'].mean() * 100
    
    # Calculate class-specific accuracy
    class_accuracy = combined.groupby('Label')['Correct'].mean() * 100  

    misclassification_matrix = pd.crosstab(combined['Label'], combined['Prediction'], normalize='index') * 100

    # Extract values for NR, R, and W
    nr_value = misclassification_matrix.at['NR', 'NR'] if 'NR' in misclassification_matrix.index and 'NR' in misclassification_matrix.columns else 0
    r_value = misclassification_matrix.at['R', 'R'] if 'R' in misclassification_matrix.index and 'R' in misclassification_matrix.columns else 0
    w_value = misclassification_matrix.at['W', 'W'] if 'W' in misclassification_matrix.index and 'W' in misclassification_matrix.columns else 0

    return overall_accuracy, class_accuracy, misclassification_matrix, nr_value, r_value, w_value

# Setup the Excel writer
folder_path = r"C:\Users\geosaad\Desktop\Main-Scripts\SpindleModelWeights_compare\Spindle-Prediction-Compare\MM_25Hz-50Hz_0.5Hz-24HzEEG"
excel_path = os.path.join(folder_path, "model_comparison_results.xlsx")
prediction_files = glob.glob(os.path.join(folder_path, '*_predictions.csv'))

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    average_across_files = []
    combined_confusion_matrix = None
    for prediction_file in prediction_files:
        base_name = os.path.splitext(os.path.basename(prediction_file))[0].replace('_predictions', '')
        label_file = os.path.join(folder_path, base_name + '.csv')
        if os.path.exists(label_file):
            sheet_name = base_name  # Sheet name based on file base name
            overall_accuracy, class_accuracy, misclassification_matrix, nr_value, r_value, w_value = compare_predictions_and_labels(prediction_file, label_file, writer, sheet_name)
            # Compute the confusion matrix for the current file
            # Accumulate the confusion matrices

            print("Confusion matrix for {}: \n{}".format(sheet_name, misclassification_matrix))

            if combined_confusion_matrix is None:
                combined_confusion_matrix = misclassification_matrix
            else:
                combined_confusion_matrix += misclassification_matrix

            plot_confusion_matrix(misclassification_matrix, f"Confusion Matrix for {sheet_name}")
            # plot_class_accuracy(class_accuracy, f"Class Accuracy for {sheet_name}")

            average_across_files.append(overall_accuracy)
            print("Overall Accuracy: {:.2f}%".format(overall_accuracy))
            print("Class Accuracy: {}".format(class_accuracy))    
            # Analyze misclassifications (False Positives)            
            # Write to Excel
            results = pd.DataFrame({
                'Overall Accuracy': [overall_accuracy],
                'Class Accuracy': [class_accuracy.to_dict()],
                'Misclassification Matrix': [misclassification_matrix.to_dict()],
                'NR Value': [nr_value],
                'R Value': [r_value],
                'W Value': [w_value]
            })
            
            results.to_excel(writer, sheet_name=sheet_name, index=False)
    if combined_confusion_matrix is not None:
        # Normalize the combined confusion matrix again to ensure it's a proper percentage
        combined_confusion_matrix = combined_confusion_matrix.div(combined_confusion_matrix.sum(axis=1), axis=0) * 100
        plot_confusion_matrix(combined_confusion_matrix, "Combined Confusion Matrix Across All Files")

    if average_across_files:
        average_accuracy = sum(average_across_files) / len(average_across_files)
        print(f"Average Accuracy across all files: {average_accuracy:.2f}%")
        
        # Write this average to a summary sheet in the Excel workbook
        pd.DataFrame({'Average Accuracy': [average_accuracy]}).to_excel(writer, sheet_name='Summary', index=False)
        
        # Optionally plot the average accuracy as a simple bar chart
        plt.figure(figsize=(6, 4))
        plt.bar("Average Accuracy", average_accuracy, color='skyblue')
        plt.title("Average Accuracy Across All Files")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.show()

# define a function that visually compares incorrect/mismatched predictions with the actual labels, color-coding the mismatches on a plot

def plot_mismatches(prediction_file, label_file):
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    prediction_file_name = os.path.basename(prediction_file)
    label_file_name = os.path.basename(label_file)
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    print("Comparing {} and {}".format(prediction_file_name, label_file_name))
    
    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['Prediction'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Find the mismatches
    mismatches = combined[combined['Correct'] == False]
    
    # Plot the mismatches
    plt.figure(figsize=(12, 6))
    plt.scatter(mismatches.index, mismatches['Prediction'], color='red', label='Predicted')
    plt.scatter(mismatches.index, mismatches['Label'], color='blue', label='Actual')
    plt.title(f"Mismatches between Predictions and Labels for {prediction_file_name} vs {label_file_name}")
    plt.xlabel('Index')
    plt.ylabel('Class')
    plt.legend()
    plt.show()
