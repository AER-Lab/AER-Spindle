import os
import pandas as pd

def process_predictions(folder_path):
    """
    Processes CSV files in the given folder that end with '_predictions-correct.csv'.

    For each file:
        - Counts the total frequency of each state (W, NR, R).
        - Computes the percentage of each state.
        - Computes bouts: counts and durations. A bout is defined as a sequence of identical states
        that are contiguous. Each row represents 4 seconds.

    The function prints the results for each file.

    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
    """
    # List all files in the folder ending with _predictions-correct.csv
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('_predictions-correct-dummy1.csv')]

    results = []  # To store analysis summary for each file

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(file_path, header=None, names=['Time', 'State'])
        states = df['State'].tolist()
        total_points = len(states)
        
        # Frequency count for states W, NR, R
        frequency = {state: states.count(state) for state in ['W', 'NR', 'R']}
        
        # Percentage calculation
        percentages = {state: (frequency[state] / total_points * 100) if total_points > 0 else 0 
                        for state in ['W', 'NR', 'R']}
        
        # Calculate bouts: each bout is a tuple (state, count)
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
            
        # For each bout, calculate duration in seconds (each row = 4 seconds)
        bout_durations = [(state, count * 4) for state, count in bouts]
        total_bouts = len(bouts)
        
        # Printing results for this file
        print(f"Results for {csv_file}:")
        print("Frequencies:")
        for state in ['W', 'NR', 'R']:
            print(f"  {state}: {frequency[state]}")
        print("Percentages:")
        for state in ['W', 'NR', 'R']:
            print(f"  {state}: {percentages[state]:.2f}%")
        print(f"Total bouts: {total_bouts}")
        print("Bout durations (seconds):")
        for idx, (state, duration) in enumerate(bout_durations, 1):
            print(f"  Bout {idx}: State: {state}, Duration: {duration} seconds")
        print("-" * 40)

        # Analyze bouts: count bouts per state and total duration per state
        bout_counts = {state: 0 for state in ['W', 'NR', 'R']}
        bout_total_duration = {state: 0 for state in ['W', 'NR', 'R']}
        
        for state, duration in bout_durations:
            bout_counts[state] += 1
            bout_total_duration[state] += duration
        
        print("Bout analysis:")
        for state in ['W', 'NR', 'R']:
            print(f"  State {state}: {bout_counts[state]} bouts, Total duration: {bout_total_duration[state]} seconds")
        print("=" * 40)

        # Append results for CSV export
        results.append({
            'File': csv_file,
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
            'R_total_duration': bout_total_duration['R']
        })

    # Export the summary results to a CSV file in the same folder
    summary_df = pd.DataFrame(results)
    output_csv_path = os.path.join(folder_path, "analysis_results.csv")
    summary_df.to_csv(output_csv_path, index=False)
    print(f"Summary CSV exported to: {output_csv_path}")

process_predictions(r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\test\ada2\ada_020625\sleep_architecture')