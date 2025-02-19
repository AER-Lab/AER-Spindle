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

def process_predictions_with_segments(folder_path):
    """Process sleep predictions with 2-hour segment analysis"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('pre-954.csv')]
    full_results = []
    segment_results = []
    
    ROWS_PER_2H = 1800  # 7200 seconds / 4 seconds per row
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, header=None, names=['Time', 'State'])
        states = df['State'].tolist()
        
        # Analyze full recording
        full_analysis = analyze_sleep_data(states, f"{csv_file}")
        full_analysis['File'] = csv_file
        full_results.append(full_analysis)
        
        # Analyze 2-hour segments
        num_segments = ceil(len(states) / ROWS_PER_2H)
        for i in range(num_segments):
            start_idx = i * ROWS_PER_2H
            end_idx = min((i + 1) * ROWS_PER_2H, len(states))
            segment_states = states[start_idx:end_idx]
            
            if len(segment_states) > 0:  # Only analyze if segment has data
                segment_analysis = analyze_sleep_data(segment_states, 
                                                    f"{csv_file}_segment_{i+1}")
                segment_analysis['File'] = csv_file
                segment_analysis['Segment_number'] = i + 1
                segment_analysis['Start_hour'] = i * 2
                segment_analysis['End_hour'] = min((i + 1) * 2, len(states) * 4 / 3600)
                segment_results.append(segment_analysis)
    
    # Create Excel file with multiple sheets
    output_excel = os.path.join(folder_path, "sleep_analysis_with_segments.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Full recording results
        pd.DataFrame(full_results).to_excel(writer, sheet_name='Full_Recording', index=False)
        # 2-hour segment results
        pd.DataFrame(segment_results).to_excel(writer, sheet_name='2h_Segments', index=False)
    
    print(f"Analysis exported to: {output_excel}")

# Run the analysis
process_predictions_with_segments(r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\Test-Hz_Comparisons_SanityCheck\ada2\ada_020625\sleep_architecture')
def compare_sleep_data(folder_path):
    """Compare sleep data between pre-954.csv and pre-954_predictions-correct.csv"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('pre-954.csv') or f.endswith('pre-954_predictions-correct.csv')]
    comparison_results = []
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, header=None, names=['Time', 'State'])
        states = df['State'].tolist()
        
        # Analyze full recording
        analysis = analyze_sleep_data(states, f"{csv_file}")
        analysis['File'] = csv_file
        comparison_results.append(analysis)
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame(comparison_results)
    
    # Create Excel file with comparison sheet
    output_excel = os.path.join(folder_path, "sleep_data_comparison.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
    
    print(f"Comparison exported to: {output_excel}")

# Run the comparison
compare_sleep_data(r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\Test-Hz_Comparisons_SanityCheck\ada2\ada_020625\sleep_architecture')