# AER Lab Documentation - Spindle Method for Scoring Sleep States in Mice

## Introduction
The AER Lab has developed a software tool for scoring sleep states in mice using a customized version of the Spindle method. 
This version will read a folder with equal EDF files [EEG/EMG] to CSV files [Time/Label] and convert it to a data set. This dataset will be fed automatically into the model based on the given shape. the model will be trained using the same parameters as SPINDLE, saving the model weights, loading the model weights and is then used to predict a given EDF producing a CSV file of the predictions. Lastly an compare/evaluation tool is provided to calculate the performance of the model across predicted files.
The implementation of this customized version was influenced and inspired by the work done in the [amibeuret/spindle](https://github.com/amibeuret/spindle) repository.

This documentation provides an overview of the AER Lab and instructions on how to use it effectively.

## Installation
To install the AER Lab, follow these steps:

1. Clone the AER Lab repository from GitHub: `git clone https://github.com/aer-lab/aer-lab.git`
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Set up the configuration file with the necessary parameters.
4. Ensure you have the right number of Epochs-to-labels
5. CSV files should not have headers

## Usage
To use the AER Lab for scoring sleep states in mice, follow these steps:

1. Prepare the input data in the required format.
2. Run the AER Lab script: `python aer_lab.py --input <input_file> --output <output_file>`
3. The AER Lab will process the input data and generate the output file with the sleep state scores.

## Configuration
The AER Lab provides a configuration file (`config.yaml`) where you can customize various parameters. Some of the important parameters include:

- `sampling_rate`: The sampling rate of the input data.
- `spindle_threshold`: The threshold value for detecting spindles.
- `sleep_state_labels`: The labels for different sleep states.

## Examples
Here are some examples of how to use the AER Lab:

- Example 1: `python aer_lab.py --input data/mouse1.csv --output results/mouse1_scores.csv`
- Example 2: `python aer_lab.py --input data/mouse2.csv --output results/mouse2_scores.csv`

## Conclusion
The AER Lab is a powerful tool for scoring sleep states in mice using the Spindle method. By following the instructions provided in this documentation, you can effectively use the AER Lab for your research or analysis.

## References
- [Spindle method for sleep scoring](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/)
- [AER Lab GitHub repository](https://github.com/aer-lab/aer-lab)