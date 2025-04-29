import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages




def compute_confusion_matrix(data, labels):
    # Assuming `data` is the predictions and `labels` are the actual labels
    return pd.crosstab(labels, data, normalize='index') * 100

def plot_confusion_matrix(conf_matrix, title, pdf_pages):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    pdf_pages.savefig()  # Save the current figure to the PDF
    # plt.show()

def compare_predictions_and_labels(prediction_file, label_file):
    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Time', 'State'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    prediction_file_name = os.path.basename(prediction_file)
    label_file_name = os.path.basename(label_file)
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    print("Comparing {} and {}".format(prediction_file_name, label_file_name))
    
    # Combine predictions and labels into a single DataFrame for easier analysis
    combined = pd.DataFrame({'Prediction': predictions['State'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']
    
    # Calculate overall accuracy
    overall_accuracy = combined['Correct'].mean() * 100
    
    # Calculate class-specific accuracy
    class_accuracy = combined.groupby('Label')['Correct'].mean() * 100  

    # Create both percentage and frequency matrices
    misclassification_matrix_pct = pd.crosstab(combined['Label'], combined['Prediction'], normalize='index') * 100
    misclassification_matrix_freq = pd.crosstab(combined['Label'], combined['Prediction'])

    return overall_accuracy, class_accuracy, misclassification_matrix_pct, misclassification_matrix_freq

def define_folder_path(folder_path):
    folder_path = folder_path
    excel_path = os.path.join(folder_path, "model_comparison_results.xlsx")
    prediction_files = glob.glob(os.path.join(folder_path, '*_predictions-correct.csv'))
    output_excel_path = os.path.join(folder_path, "mismatches_summary.xlsx")
    return folder_path, excel_path, prediction_files, output_excel_path

def compare_files(folder_path):
    folder_path, excel_path, prediction_files, output_excel_path = define_folder_path(folder_path)
    matrix_pdf_pages = PdfPages(os.path.join(folder_path, 'Matrix_plots.pdf'))

    print("Prediction files are: ", prediction_files)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        average_across_files = []
        combined_confusion_matrix = None
        for prediction_file in prediction_files:
            print("Processing file: {}".format(prediction_file))
            base_name = os.path.splitext(os.path.basename(prediction_file))[0].replace('_predictions-correct', '')
            print("Basename: ", base_name)
            label_file = os.path.join(folder_path, base_name + '.csv')
            print("prediction file and label file: ", prediction_file, label_file)
            if os.path.exists(label_file):
                sheet_name = base_name  # Sheet name based on file base name
                overall_accuracy, class_accuracy, misclassification_matrix, misclassification_matrix_freq= compare_predictions_and_labels(prediction_file, label_file)
                
                print("Confusion matrix for {}: \n{}".format(sheet_name, misclassification_matrix))

                if combined_confusion_matrix is None:
                    combined_confusion_matrix = misclassification_matrix
                else:
                    combined_confusion_matrix += misclassification_matrix

                plot_confusion_matrix(misclassification_matrix, f"Confusion Matrix for {sheet_name}", matrix_pdf_pages)
                plot_confusion_matrix(misclassification_matrix_freq, f"Confusion Matrix for {sheet_name} (Frequency)", matrix_pdf_pages)

                average_across_files.append(overall_accuracy)
                print("Overall Accuracy: {:.2f}%".format(overall_accuracy))
                print("Class Accuracy: {}".format(class_accuracy))    
                
                # Create detailed results DataFrame
                results = pd.DataFrame({
                })

                # Write results to Excel
                results.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                
                # Add percentage confusion matrix
                writer.sheets[sheet_name].cell(row=1, column=1, value="Confusion Matrix (Percentages)")
                misclassification_matrix.to_excel(writer, sheet_name=sheet_name, startrow=1)
                
                # Add frequency confusion matrix
                writer.sheets[sheet_name].cell(row=8, column=1, value="Confusion Matrix (Frequencies)")
                misclassification_matrix_freq.to_excel(writer, sheet_name=sheet_name, startrow=8)
        if combined_confusion_matrix is not None:
            # Normalize the combined confusion matrix again to ensure it's a proper percentage
            combined_confusion_matrix = combined_confusion_matrix.div(combined_confusion_matrix.sum(axis=1), axis=0) * 100
            # plot_confusion_matrix(combined_confusion_matrix, "Combined Confusion Matrix Across All Files", matrix_pdf_pages)
    matrix_pdf_pages.close()
    loop_files_to_compare(folder_path, output_excel_path)

def plot_mismatches(prediction_file, label_file, pdf_pages):
    """
    Plots mismatches between prediction and label files in 2-hour segments.
    """

    print(f"Comparing {os.path.basename(prediction_file)} with {os.path.basename(label_file)}")

    # Load the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Time', 'State'])
    labels = pd.read_csv(label_file, header=None, names=['Time', 'Label'])
    
    # Normalize label states
    labels['Label'] = labels['Label'].replace(regex={r'W.*': 'W', r'R.*': 'R', r'NR.*': 'NR'})

    # Combine predictions and labels
    combined = pd.DataFrame({'Prediction': predictions['State'], 'Label': labels['Label']})
    combined['Correct'] = combined['Prediction'] == combined['Label']

    # Determine the number of plots needed
    num_rows = len(combined)
    rows_per_plot = 2 * 3600 // 4  # 2 hours of data, assuming each row is 4 seconds
    num_plots = (num_rows // rows_per_plot) + 1
    label_file_name = os.path.basename(label_file).replace('.csv', '').capitalize()
    for i in range(num_plots):
        start_row = i * rows_per_plot
        end_row = min((i + 1) * rows_per_plot, num_rows)

        # Plot the predictions and labels for the current segment
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_row, end_row), combined['Prediction'][start_row:end_row].reset_index(drop=True).astype(str), label='Prediction', color='blue')
        plt.plot(range(start_row, end_row), combined['Label'][start_row:end_row].reset_index(drop=True).astype(str), label='Label', color='green')

        plt.title(f"{(label_file_name)} - Predictions vs Labels (Segment {i + 1})")
        plt.xlabel("Epoch # (2-hour segments)")
        plt.ylabel("Sleep Stage")
        plt.legend()
        pdf_pages.savefig()  # Save the current figure to the PDF
        plt.close()
    
def loop_files_to_compare(folder_path, output_excel_path):
    """
    Loops through files in the folder, compares prediction and label files, 
    and writes mismatch data to an Excel file.
    """
    mismatch_pdf_pages = PdfPages(os.path.join(folder_path, 'Mismatch_plots.pdf'))

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    prediction_files = glob.glob(os.path.join(folder_path, '*_predictions-correct.csv'))
    for label_file in csv_files:
        prediction_file = label_file.replace('.csv', '_predictions-correct.csv')
        if prediction_file in prediction_files:
            plot_mismatches(prediction_file, label_file, mismatch_pdf_pages)
    mismatch_pdf_pages.close()  # Close the PDF file



