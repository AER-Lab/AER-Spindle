import os
import pandas as pd
from math import ceil

def analyze_sleep_data(states, segment_name=""):
    """Helper function to analyze a segment of sleep states"""
    total_points = len(states)
    
    # Frequency and percentage calculation
    frequency = {state: states.count(state) for state in ['W', 'NR', 'R']}
    percentages = {state: (frequency[state] / total_points * 100) if total_points > 0 else 0 
                    for state in ['W', 'NR', 'R']}
    
    # Calculate bouts
    bouts = []
    if states:
        current_state = states[0]
        count = 1
        for s in states[1:]:
            if s == current_state:
                count += 1
            else:
                bouts.append((current_state, count))
                current_state = s
                count = 1
        bouts.append((current_state, count))
        
    bout_durations = [(state, count * 4) for state, count in bouts]
    total_bouts = len(bouts)
    
    # Analyze bouts per state
    bout_counts = {state: 0 for state in ['W', 'NR', 'R']}
    bout_total_duration = {state: 0 for state in ['W', 'NR', 'R']}
    
    for state, duration in bout_durations:
        bout_counts[state] += 1
        bout_total_duration[state] += duration
    
    mean_bout_duration = {state: bout_total_duration[state] / bout_counts[state] 
                         if bout_counts[state] > 0 else 0
                         for state in ['W', 'NR', 'R']}
    
    return {
        'Segment': segment_name,
        'W_freq': frequency['W'],
        'NR_freq': frequency['NR'],
        'R_freq': frequency['R'],
        'W_percent': percentages['W'],
        'NR_percent': percentages['NR'],
        'R_percent': percentages['R'],
        'Total_bouts': total_bouts,
        'W_bout_count': bout_counts['W'],
        'NR_bout_count': bout_counts['NR'],
        'R_bout_count': bout_counts['R'],
        'W_total_duration': bout_total_duration['W'],
        'NR_total_duration': bout_total_duration['NR'],
        'R_total_duration': bout_total_duration['R'],
        'W_mean_duration': mean_bout_duration['W'],
        'NR_mean_duration': mean_bout_duration['NR'],
        'R_mean_duration': mean_bout_duration['R']
    }

def compare_sleep_data(folder_path):
    """Compare sleep data between original files and their predictions"""
    # Find all prediction files and match with originals
    prediction_files = [f for f in os.listdir(folder_path) 
                       if f.endswith('_predictions-correct.csv')]
    file_pairs = []
    for pred_file in prediction_files:
        orig_file = pred_file.replace('_predictions-correct.csv', '.csv')
        if orig_file in os.listdir(folder_path):
            file_pairs.append((orig_file, pred_file))
        else:
            print(f"Warning: Original file {orig_file} not found for {pred_file}")
    
    full_comparison_results = []
    segment_comparison_results = []
    segment_data = {}
    
    ROWS_PER_2H = 1800  # 7200 seconds / 4 seconds per row
    
    # Process each file pair
    for orig_file, pred_file in file_pairs:
        for csv_file in [orig_file, pred_file]:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path, header=None, names=['Time', 'State'])
            main_data = df['State'].tolist()
            
            # Analyze full recording
            full_analysis = analyze_sleep_data(main_data, f"{csv_file}")
            full_analysis['File'] = csv_file
            full_comparison_results.append(full_analysis)
            
            # Analyze 2-hour segments
            num_segments = ceil(len(main_data) / ROWS_PER_2H)
            for i in range(num_segments):
                start_idx = i * ROWS_PER_2H
                end_idx = min((i + 1) * ROWS_PER_2H, len(main_data))
                corrected_data = main_data[start_idx:end_idx]
                
                if len(corrected_data) > 0:
                    segment_analysis = analyze_sleep_data(corrected_data, 
                                                        f"{csv_file}_segment_{i+1}")
                    segment_analysis['File'] = csv_file
                    segment_analysis['Segment_number'] = i + 1
                    segment_analysis['Start_hour'] = i * 2
                    segment_analysis['End_hour'] = min((i + 1) * 2, len(main_data) * 4 / 3600)
                    segment_comparison_results.append(segment_analysis)
                    
                    segment_key = f"Segment_{i+1}"
                    if segment_key not in segment_data:
                        segment_data[segment_key] = []
                    segment_data[segment_key].append(segment_analysis)
    
    # Create Excel file with comparison sheets
    output_excel = os.path.join(folder_path, "sleep_data_comparison.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(full_comparison_results).to_excel(writer, 
                                                     sheet_name='Full_Comparison', 
                                                     index=False)
        pd.DataFrame(segment_comparison_results).to_excel(writer, 
                                                        sheet_name='2h_Segment_Comparison',
                                                        index=False)
        for segment_key, data in segment_data.items():
            pd.DataFrame(data).to_excel(writer, sheet_name=segment_key, index=False)
    
    print(f"Comparison exported to: {output_excel}")

# Run the comparison
compare_sleep_data(r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\Test-Hz_Comparisons_SanityCheck\ada2\ada_020625\sleep_architecture\4-files')
