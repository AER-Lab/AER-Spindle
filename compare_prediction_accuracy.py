import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import openpyxl
from openpyxl.formatting.rule import FormulaRule

# Setup the Excel writer
folder_path = r"C:\Users\Public\Documents"
excel_path = os.path.join(folder_path, "model_comparison_results.xlsx")
prediction_files = glob.glob(os.path.join(folder_path, '*_predictions.csv'))
output_excel_path = os.path.join(folder_path, "mismatches_summary.xlsx")



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
    predictions = pd.read_csv(prediction_file, header=None, names=['Epoch #', 'Prediction'])
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


def compare_files():
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
    loop_files_to_compare(folder_path)






def plot_mismatches(prediction_file, label_file):
    print("Prediction file: ", prediction_file)
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Epoch','Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    prediction_file_name = os.path.basename(prediction_file)
    label_file_name = os.path.basename(label_file)
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    print("Comparing {} and {}".format(prediction_file_name, label_file_name))
    
    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['Prediction'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Determine the number of rows and calculate the number of plots needed
    num_rows = len(combined)
    rows_per_plot = 2 * 3600 // 4  # 2 hours worth of data, assuming each row is 4 seconds
    num_plots = (num_rows // rows_per_plot) + 1
    
    for i in range(num_plots):
        start_row = i * rows_per_plot
        end_row = min((i + 1) * rows_per_plot, num_rows)
        
        # Plot the predictions and labels for the current segment
        plt.figure(figsize=(12, 6))
        plt.plot(combined['Prediction'][start_row:end_row].reset_index(drop=True), label='Prediction', color='blue')
        plt.plot(combined['Label'][start_row:end_row].reset_index(drop=True), label='Label', color='green')
        
        # Add vertical lines where there are mismatches
        mismatches = combined[start_row:end_row][combined['Correct'] == False].index - start_row
        # for mismatch in mismatches:
        #     plt.axvline(x=mismatch, color='red', linestyle='--', linewidth=0.5)
        
        plt.title(f"Predictions vs Labels with Mismatches (Segment {i + 1})")
        plt.xlabel("Time (in 2-hour segments)")
        plt.ylabel("Stage")
        plt.legend()
        plt.show()

# Example usage
os.chdir(folder_path)

def save_mismatches_to_excel(prediction_file, label_file, output_file):
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['Prediction'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Find the mismatches
    mismatches = combined[combined['Correct'] == False]
    
    # Save mismatches to an Excel file with highlighting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        mismatches.to_excel(writer, sheet_name='Mismatches', index=True)
        workbook = writer.book
        worksheet = writer.sheets['Mismatches']
        
        # Apply conditional formatting to highlight mismatches
        from openpyxl.formatting.rule import FormulaRule
        yellow_fill = openpyxl.styles.PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        for row in range(2, len(mismatches) + 2):
            worksheet.conditional_formatting.add(f'C{row}:D{row}', 
                FormulaRule(formula=[f'$C{row}<>$D{row}'], fill=yellow_fill))

def analyze_mismatches(prediction_file, label_file, output_file):
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['Prediction'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Find the mismatches
    mismatches = combined[combined['Correct'] == False]
    
    # Calculate frequency and percentage of mismatches for each class
    mismatch_counts = mismatches['Label'].value_counts()
    total_counts = combined['Label'].value_counts()
    
    # Calculate basic statistics
    overall_accuracy = combined['Correct'].mean() * 100
    
    # Perform t-test to see if there's a significant difference between the classes
    nr_errors = mismatches[mismatches['Label'] == 'NR'].shape[0]
    r_errors = mismatches[mismatches['Label'] == 'R'].shape[0]
    w_errors = mismatches[mismatches['Label'] == 'W'].shape[0]
    t_stat, p_value = ttest_ind([nr_errors, r_errors, w_errors], [total_counts['NR'], total_counts['R'], total_counts['W']])
    
    # Calculate quarterly percentile score of errors
    mismatches = mismatches.copy()  # Create a copy to avoid the SettingWithCopyWarning
    mismatches.loc[:, 'Quarter'] = pd.qcut(mismatches.index, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    quarterly_errors = mismatches['Quarter'].value_counts().sort_index()
    
    # Save mismatches and statistics to an Excel file with highlighting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        mismatches.to_excel(writer, sheet_name='Mismatches', index=True)
        worksheet = writer.sheets['Mismatches']
        
        # Apply conditional formatting to highlight mismatches
        for row in range(2, len(mismatches) + 2):
            yellow_fill = openpyxl.styles.PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            worksheet.conditional_formatting.add(f'C{row}:D{row}', 
                FormulaRule(formula=[f'$C{row}<>$D{row}'], fill=yellow_fill))
        
        # Save statistics to another sheet
        stats = pd.DataFrame({
            'Overall Accuracy': [overall_accuracy],
            'NR Errors': [nr_errors],
            'R Errors': [r_errors],
            'W Errors': [w_errors],
            'T-Statistic': [t_stat],
            'P-Value': [p_value]
        })
        stats.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Save quarterly errors to another sheet
        quarterly_errors.to_excel(writer, sheet_name='Quarterly Errors', index=True)



def loop_files_to_compare(folder_path):
    # first check for .csv and _predictions.csv files
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    prediction_files = glob.glob(os.path.join(folder_path, '*_predictions.csv'))
    for file_to_compare in csv_files:
        prediction_file = file_to_compare.replace('.csv', '_predictions.csv')
        if prediction_file in prediction_files:
            plot_mismatches(prediction_file, file_to_compare)
            save_mismatches_to_excel(prediction_file, file_to_compare, output_excel_path)
            analyze_mismatches(prediction_file, file_to_compare, output_excel_path)

