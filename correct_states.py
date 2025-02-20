import pandas as pd
import os

def correct_states(df):
    # Run the correction process three times to ensure propagation of changes
    for _ in range(3):
        i = 0
        while i < len(df):
            #print(f"Checking i={i}, State={df.loc[i, 'State']}")

            # Correct short sequences of 'R' (1-3 episodes) flanked by 'W'
            if df.loc[i, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'W':
                rem_count = 1
                j = i
                while j + 1 < len(df) and df.loc[j + 1, 'State'] == 'R' and rem_count < 4:
                    j += 1
                    rem_count += 1

                if j + 1 < len(df) and df.loc[j + 1, 'State'] == 'W':
                    df.loc[i:j+1, 'State'] = 'W'
                    i += 1  # Move to the next index
                    #print(f"Corrected W to R at positions {i} to {end_w}")


            # Check for long NR (>5) -> short W (<5) -> long R (>5) transition -- changing W to NR
            if i < len(df) - 1 and df.loc[i, 'State'] == 'W' and df.loc[i + 1, 'State'] == 'R':
                # Identify if W preceding 'R' is shorter than 5
                start_w = i
                end_w = i
                start_r = i
                while start_w > 0 and df.loc[start_w - 1, 'State'] == 'W':
                    start_w -= 1
                w_length = end_w - start_w + 1
                if w_length < 5:
                    # Identify if NR preceding W is longer than 5
                    end_nr = start_w - 1
                    start_nr = start_w - 1
                    nr_length = 0  # Initialize NR length
                    while start_nr > 0 and df.loc[start_nr - 1, 'State'] == 'NR':
                        nr_length += 1
                        start_nr -= 1
                        if nr_length > 5:
                            break  # Stop counting as NR length exceeds 5
                    nr_length += 1  # Include the initial NR at start_w - 1

                    # Identify if R following W is longer than 5
                    start_r = end_w + 1
                    end_r = end_w + 1
                    r_length = 0  # Initialize R length
                    while end_r < len(df) - 1 and df.loc[end_r + 1, 'State'] == 'R':
                        r_length += 1
                        end_r += 1
                        if r_length > 5:
                            break  # Stop counting as R length exceeds 5
                    r_length += 1  # Include the initial R at end_w + 1

                    # If both NR and R are long enough, correct W to NR
                    if nr_length > 5 and r_length > 5:
                        df.loc[start_w:end_w, 'State'] = 'NR'
                        i += 1  # Move index to the next epoch

                    # If NR is long enough, but R is short, correct R to W
                    if nr_length > 5 and r_length < 5:
                        df.loc[start_r:end_r, 'State'] = 'W'
                        i += 1

            #Correct additional errors:

            # Correct single 'NR' preceded by 'R' and followed by 'W' (R-NR-W = R-W-W)
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'NR' and df.loc[i - 1, 'State'] == 'R' and df.loc[i + 1, 'State'] == 'W':
                df.loc[i, 'State'] = 'W'
                i += 1

            # Correct single 'NR' surrounded by 'W' (W-NR-W = W-W-W)
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'NR' and df.loc[i - 1, 'State'] == 'W' and df.loc[i + 1, 'State'] == 'W':
                df.loc[i, 'State'] = 'W'
                i += 1

            # Correct single 'NR' surrounded by at least two 'R' on both sides (R-R-NR-R-R = R-R-R-R-R)
            if i > 1 and i < len(df) - 2 and df.loc[i, 'State'] == 'NR':
                # Check if the two states before and after the 'NR' are 'R'
                if all(df.loc[i - j, 'State'] == 'R' for j in range(1, 3)) and all(df.loc[i + k, 'State'] == 'R' for k in range(1, 3)):
                    df.loc[i, 'State'] = 'R'
                    i += 1

            # Correct single 'W' surrounded by at least three 'R' on both sides (R-R-R-W-R-R-R = R-R-R-R-R-R-R)
            if i > 2 and i < len(df) - 3 and df.loc[i, 'State'] == 'W':
                # Check if there are at least three 'R' before and after the 'W'
                if all(df.loc[i - j, 'State'] == 'R' for j in range(1, 4)) and all(df.loc[i + k, 'State'] == 'R' for k in range(1, 4)):
                    df.loc[i, 'State'] = 'R'
                    i += 1

            # Correct single 'R' preceded by 'W' and followed by 'NR' (W-R-NR = W-W-NR)
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'W' and df.loc[i + 1, 'State'] == 'NR':
                df.loc[i, 'State'] = 'W'
                i += 1

            # Correct double 'R' preceded by 'W' and followed by 'NR' (W-R-R-NR = W-W-W-NR)
            if i > 1 and i < len(df) - 2 and df.loc[i, 'State'] == 'R' and df.loc[i + 1, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'W' and df.loc[i + 2, 'State'] == 'NR':
                df.loc[i, 'State'] = 'W'
                df.loc[i + 1, 'State'] = 'W'
                i += 1

            # Correct single 'R' preceded by 'NR' and followed by 'W' (NR-R-W = NR-W-W)
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'NR' and df.loc[i + 1, 'State'] == 'W':
                df.loc[i, 'State'] = 'W'
                i += 1

            # Correct single 'R' surrounded by 'NR' (NR-R-NR = NR-NR-NR)
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'NR' and df.loc[i + 1, 'State'] == 'NR':
                df.loc[i, 'State'] = 'NR'
                i += 1

            # Correct double 'R' surrounded by 'NR' (NR-R-R-NR = NR-NR-NR-NR)
            if i > 0 and i < len(df) - 2 and df.loc[i, 'State'] == 'R' and df.loc[i + 1, 'State'] == 'R' and df.loc[i - 1, 'State'] == 'NR' and df.loc[i + 2, 'State'] == 'NR':
                df.loc[i, 'State'] = 'NR'
                df.loc[i + 1, 'State'] = 'NR'
                i += 1

            # If one want to remove single epoches of W during NR, uncomment the below. 
            # Correct single 'W' surrounded by 'NR' (NR-W-NR = NR-NR-NR)
            # TODO: UNCOMMENT THIS BLOCK IF YOU WANT TO REMOVE SINGLE EPOCHS OF W DURING NR
            if i > 0 and i < len(df) - 1 and df.loc[i, 'State'] == 'W' and df.loc[i - 1, 'State'] == 'NR' and df.loc[i + 1, 'State'] == 'NR':
                df.loc[i, 'State'] = 'NR'
                i += 1
                
            # Correct 3-4 episodes of 'R' preceded by at least 4 'NR' and followed by at least 2 'NR'
            if i > 3 and i < len(df) - 4:
                # Check if current epoch is 'R'
                if df.loc[i, 'State'] == 'R':
                    # Find consecutive 'R' sequence
                    start_r = i
                    end_r = i
                    while end_r + 1 < len(df) and df.loc[end_r + 1, 'State'] == 'R':
                        end_r += 1
                    
                    r_length = end_r - start_r + 1
                    
                    # Check if R sequence is 3-4 epochs long
                    if 3 <= r_length <= 4:
                        # Check for 4 'NR' before
                        if all(df.loc[start_r - k, 'State'] == 'NR' for k in range(1, 5)):
                            # Check for 2 'NR' after
                            if all(df.loc[end_r + k, 'State'] == 'NR' for k in range(1, 5)):
                                # Convert R sequence to NR
                                df.loc[start_r:end_r, 'State'] = 'NR'
                                i = end_r + 1
                                continue
                            
                            
            # Correct 2-3 episodes of 'NR' surrounded by at least 4 'R' on both sides
            if i > 3 and i < len(df) - 4:
                # Check if current epoch is 'NR'
                if df.loc[i, 'State'] == 'NR':
                    # Find consecutive 'NR' sequence
                    start_nr = i
                    end_nr = i
                    while end_nr + 1 < len(df) and df.loc[end_nr + 1, 'State'] == 'NR':
                        end_nr += 1
                    
                    nr_length = end_nr - start_nr + 1
                    
                    # Check if NR sequence is 2-3 epochs long
                    if nr_length == 2 or nr_length == 3:
                        # Check for at least 4 'R' before
                        if all(df.loc[start_nr - k, 'State'] == 'R' for k in range(1, 5)):
                            # Check for at least 4 'R' after
                            if all(df.loc[end_nr + k, 'State'] == 'R' for k in range(1, 5)):
                                # Convert NR sequence to R
                                df.loc[start_nr:end_nr, 'State'] = 'R'
                                i = end_nr + 1
                                continue  # Skip to the next epoch after the corrected sequence

            i += 1

    return df


def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('_predictions.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), header=None)
            df.columns = ['Time', 'State']
            df_corrected = correct_states(df)
            df_corrected.to_csv(os.path.join(output_dir, file.replace('.csv', '-correct.csv')), header=False, index=False)
            print(f"Processed: {file}")


if __name__ == '__main__':
    input_dir = r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\Test-Hz_Comparisons_SanityCheck\ada2\ada_020625\sleep_architecture'
    output_dir = r'C:\Users\geosaad\Desktop\Su-EEG-EDF-DATA\Test-Hz_Comparisons_SanityCheck\ada2\ada_020625\sleep_architecture'
    process_files(input_dir, output_dir)
    print("Processing complete.")