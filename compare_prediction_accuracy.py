import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import openpyxl
from openpyxl.formatting.rule import FormulaRule
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# Setup the Excel writer
folder_path = r"C:\Users\geosaad\Desktop\Main-Scripts\SpindleModelWeights_compare\Spindle-Prediction-Compare\Model_Comparison\Validate_State_Transitions"
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



def check_w_r_transitions(prediction_file, label_file):
    """
    Check for W-R transitions in the prediction file, compare with the true state in the label file,
    and export surrounding data to an Excel file with highlighting.

    Args:
        prediction_file (str): Path to the prediction CSV file.
        label_file (str): Path to the label CSV file.

    Returns:
        str: A message indicating the presence of W-R transitions and the specific rows/epochs where they occur.
    """
    # Load the predictions and labels from the CSV files
    predictions = pd.read_csv(prediction_file, header=None, names=['Epoch #', 'Prediction'])
    labels = pd.read_csv(label_file, header=None, names=['Epoch #', 'Label'])

    # Ensure both files have matching rows
    if len(predictions) != len(labels):
        return "Mismatch in the number of rows between predictions and labels."

    # Concatenate predictions and labels along the columns
    merged = pd.concat([predictions, labels['Label']], axis=1)
    # merged = pd.merge(predictions, labels, on='Epoch #', how='inner')

    w_r_transitions = []  # To store the indices of W-R transitions

    # Identify W-R transitions
    for i in range(1, len(merged)):
        current = merged['Prediction'][i]
        previous = merged['Prediction'][i - 1]

        if previous == 'W' and current == 'R':  # Detect W-R transitions
            w_r_transitions.append(i)  # Store the index of the transition

    if not w_r_transitions:
        return "No W-R transitions found in the prediction file."

    # Collect surrounding data for each transition
    surrounding_data = []
    for idx in w_r_transitions:
        start = max(0, idx - 4)  # Ensure we don't go below 0
        end = min(len(merged), idx + 5)  # Ensure we don't go past the last row
        transition_data = merged.iloc[start:end].copy()
        transition_data['Transition Highlight'] = ['Yes' if i == idx else '' for i in range(start, end)]
        surrounding_data.append(transition_data)

    # Combine all surrounding data into one DataFrame
    all_data = pd.concat(surrounding_data, ignore_index=True)

    # Create an Excel file dynamically named after the prediction file
    output_file = prediction_file.replace('.csv', '_W-R_Transitions.xlsx')

    # Initialize a new workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "W-R Transitions"

    # Add the data to the worksheet
    for row in dataframe_to_rows(all_data, index=False, header=True):
        ws.append(row)

    # Apply formatting (highlight transitions with a color and add borders)
    highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                          top=Side(style='thin'), bottom=Side(style='thin'))

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=len(all_data.columns)):
        if row[-1].value == 'Yes':  # Highlight rows where 'Transition Highlight' is 'Yes'
            for cell in row:
                cell.fill = highlight_fill
        for cell in row:
            cell.border = thin_border

    # Save the workbook
    wb.save(output_file)

    # Generate the output message
    message = f"Found {len(w_r_transitions)} W-R transitions:\n"
    message += ", ".join(f"Epoch {merged['Epoch #'][row]}" for row in w_r_transitions)
    message += f"\nData exported to {output_file} with highlighted transitions and true states included."

    return message




def compare_predictions_and_labels(prediction_file, label_file, writer, sheet_name):

    # First check if there are W-R or R-W transitions in the predictin file
    
    output_excel = "w_r_transitions.xlsx"
    result_message = check_w_r_transitions(prediction_file, label_file)
    print(result_message)


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


# Function to read _W-R_Transitions.xlsx files, count number of transitions. count how many R epoches there are after W-R transitions, count how many W epoches before W-R transitions
def correct_transitions(file_path):
    """
    Correct transitions in the W-R Transitions sheet and highlight corrections in orange, 
    and transitions in yellow.
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name="W-R Transitions")

    # Initialize counters and details list
    w_r_transitions = 0
    r_epochs_after_w_r = 0
    w_epochs_before_w_r = 0
    transition_details = []

    # Add correction and notes columns if not present
    if 'Correction' not in df.columns:
        df['Correction'] = df['Prediction']
    if 'Notes' not in df.columns:
        df['Notes'] = ''
    else:
        df['Notes'] = df['Notes'].fillna('')

    # Apply rules for W-R transitions and corrections
    for i in range(1, len(df) - 1):
        if df.iloc[i - 1]['Prediction'] == 'W' and df.iloc[i]['Prediction'] == 'R':
            w_r_transitions += 1

            # Count R epochs after W-R transition
            r_count = 0
            for j in range(i, len(df)):
                if df.iloc[j]['Prediction'] == 'R':
                    r_epochs_after_w_r += 1
                    r_count += 1
                else:
                    break

            # Count W epochs before W-R transition
            w_count = 0
            for k in range(i - 1, -1, -1):
                if df.iloc[k]['Prediction'] == 'W':
                    w_epochs_before_w_r += 1
                    w_count += 1
                else:
                    break

            # Store the details of the transition
            transition_details.append({
                'Transition Index': i,
                'W Epochs Before': w_count,
                'R Epochs After': r_count
            })

            # Add notes and corrections
            if pd.isna(df.at[i, 'Notes']):
                df.at[i, 'Notes'] = ''
            df.at[i, 'Notes'] += f'W-R transition at index {i}, W before: {w_count}, R after: {r_count}'

            # Rule 1: Correct R to W if W > 4 before and either W/NR follows R
            if w_count > 4 and (df.iloc[i + 1]['Prediction'] in ['W', 'NR'] or df.iloc[i + 2]['Prediction'] in ['W', 'NR']):
                df.at[i, 'Correction'] = 'W'
                df.at[i, 'Notes'] += ' | Rule 1: Corrected, R changed to W due to >4 W before and W/NR after'

            # Rule 2: Handle W (1-4) before R with long NR preceding W
            if (
                1 <= w_count <= 4 and
                df.iloc[max(0, k - 5):k]['Prediction'].tolist().count('NR') > 5 and
                r_count > 5
            ):
                df.loc[max(0, k - w_count):k, 'Correction'] = 'NR'
                df.at[i, 'Notes'] += ' | Rule 2: Corrected, W changed to NR due to long NR before W and >5 R after'

    # Rule 3: Single R correction (NR >3 before, W after R)
    for i in range(1, len(df) - 4):
        if (
            df.iloc[i - 3:i]['Prediction'].tolist().count('NR') >= 3 and
            df.iloc[i]['Prediction'] == 'R' and
            df.iloc[i + 1:i + 5]['Prediction'].tolist().count('W') >= 3
        ):
            df.at[i, 'Correction'] = 'W'
            df.at[i, 'Notes'] += ' | Rule 3: Corrected, Single R changed to W due to NR >3 before and W >3 after'

    # Rule 4: Single R correction with following R in 4 epochs
    for i in range(1, len(df) - 4):
        if (
            df.iloc[i - 3:i]['Prediction'].tolist().count('NR') >= 3 and
            df.iloc[i]['Prediction'] == 'R' and
            df.iloc[i + 1:i + 5]['Prediction'].tolist().count('R') > 2
        ):
            df.loc[i + 1:i + 5, 'Correction'] = 'R'
            df.loc[i + 1:i + 5, 'Notes'] += ' | Rule 4: Corrected, Following W changed to R due to >2 R after'

    # Rule 5: NR-R(single)-NR transition
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i - 1]['Prediction'] == 'NR' and
            df.iloc[i]['Prediction'] == 'R' and
            df.iloc[i + 1]['Prediction'] == 'NR'
        ):
            df.at[i, 'Correction'] = 'NR'
            df.at[i, 'Notes'] += ' | Rule 5: Corrected, Single R changed to NR'

    # Rule 6: Single NR correction (W-NR(single)-W)
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i - 1]['Prediction'] == 'W' and
            df.iloc[i]['Prediction'] == 'NR' and
            df.iloc[i + 1]['Prediction'] == 'W'
        ):
            df.at[i, 'Correction'] = 'W'
            df.at[i, 'Notes'] += ' | Rule 6: Corrected, Single NR changed to W due to W before and after'

    # Rule 7: Long R (>4) - NR(single) - W (>3)
    for i in range(6, len(df) - 4):
        if (
            df.iloc[i - 4:i]['Prediction'].tolist().count('R') > 4 and
            df.iloc[i]['Prediction'] == 'NR' and
            df.iloc[i + 1:i + 4]['Prediction'].tolist().count('W') > 3
        ):
            df.at[i, 'Correction'] = 'W'
            df.at[i, 'Notes'] += ' | Rule 7: Corrected, NR changed to W due to >4 R before and >3 W after'

    # Rule 8: Long R (>6) to NR (>6)
    for i in range(12, len(df)):
        if (
            df.iloc[i - 6:i]['Prediction'].tolist().count('R') > 6 and
            df.iloc[i:i + 6]['Prediction'].tolist().count('NR') > 6
        ):
            df.loc[i - 2:i, 'Correction'] = 'W'
            df.loc[i - 2:i, 'Notes'] += ' | Rule 8: Corrected, Artificial W inserted for long R to NR'

    # Rule 9: Correct last state in the document
    if df.iloc[-1]['Prediction'] in ['R', 'NR']:
        df.at[len(df) - 1, 'Correction'] = 'W'
        df.at[len(df) - 1, 'Notes'] += ' | Rule 9: Corrected last state to W'

    # Rule 10: Check if last 3 were W, and at the signal W-R-R, turn into W
    for i in range(2, len(df)):
        if (
            df.iloc[i - 2]['Prediction'] == 'W' and
            df.iloc[i - 1]['Prediction'] == 'R' and
            df.iloc[i]['Prediction'] == 'R'
        ):
            df.at[i - 1, 'Correction'] = 'W'
            df.at[i, 'Correction'] = 'W'
            df.at[i - 1, 'Notes'] += ' | Rule 10: Corrected, R-R changed to W-W due to W before'
            df.at[i, 'Notes'] += ' | Rule 10: Corrected, R-R changed to W-W due to W before'
    # Rule 11: If W > 4 and W-R signal and multiple R > 3 and W > 3 after, turn R to W
    for i in range(1, len(df) - 4):
        if (
            df.iloc[i - 4:i]['Prediction'].tolist().count('W') > 4 and
            df.iloc[i]['Prediction'] == 'R' and
            df.iloc[i + 1:i + 4]['Prediction'].tolist().count('R') > 3 and
            df.iloc[i + 1:i + 4]['Prediction'].tolist().count('W') > 3
        ):
            df.at[i, 'Correction'] = 'W'
            df.at[i, 'Notes'] += ' | Rule 11: Corrected, R changed to W due to >4 W before, >3 R and >3 W after'
    
    # Rule 12 : Single R's when W > 3 before, change to W
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i - 1]['Prediction'] == 'W' and
            df.iloc[i]['Prediction'] == 'R' and
            df.iloc[i + 1]['Prediction'] == 'W'
        ):
            df.at[i, 'Correction'] = 'W'
            df.at[i, 'Notes'] += ' | Rule 12: Corrected, Single R changed to W due to W before and after'
    # Save corrections directly to the W-R Transitions sheet
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, sheet_name='W-R Transitions')

    # Apply formatting
    wb = load_workbook(file_path)
    ws = wb['W-R Transitions']
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    orange_fill = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=6):
        transition_value = row[3].value  # Assuming 'Transition Highlight' is the 4th column
        notes_value = row[5].value  # Assuming 'Notes' is the 6th column
        if notes_value and 'Corrected' in notes_value:  # Highlight corrections in orange
            for cell in row:
                cell.fill = orange_fill
        elif transition_value == 'Yes':  # Highlight W-R transitions in yellow
            for cell in row:
                cell.fill = yellow_fill
        for cell in row:  # Apply thin borders to all rows
            cell.border = thin_border
    
    

    wb.save(file_path)

    return w_r_transitions, r_epochs_after_w_r, w_epochs_before_w_r, transition_details
# Example usage
# compare_files()
file_path = r'C:\Users\geosaad\Desktop\Main-Scripts\SpindleModelWeights_compare\Spindle-Prediction-Compare\Model_Comparison\Validate_State_Transitions\post-1100_predictions_W-R_Transitions.xlsx'
w_r_transitions, r_epochs_after_w_r, w_epochs_before_w_r, transition_details = correct_transitions(file_path)
print(f"Number of W-R transitions: {w_r_transitions}")
print(f"Number of R epochs after W-R transitions: {r_epochs_after_w_r}")
print(f"Number of W epochs before W-R transitions: {w_epochs_before_w_r}")

# Print detailed information for each transition
for detail in transition_details:
    print(f"Transition Index: {detail['Transition Index']}, W Epochs Before: {detail['W Epochs Before']}, R Epochs After: {detail['R Epochs After']}")