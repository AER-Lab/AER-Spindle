
# V1

import torch

class_mapping = {
    0: "W",
    1: "NR",
    2: "R"
}

def predict_sleep_stages(spectrograms, model):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    last_valid_prediction = None  # Initialize to None, assuming we don't know the first state yet
    
    with torch.no_grad():  # Disable gradient calculation
        num_datapoint = spectrograms.shape[0]  # Number of epochs
        for i in range(num_datapoint):
            epoch_spectrogram = torch.tensor(spectrograms[i]).unsqueeze(0)  # Add batch dimension
            output = model(epoch_spectrogram)
            
            # Get the predicted class (assuming output is logits, apply softmax if necessary)
            predicted_class = output.argmax(dim=1).item()
            if predicted_class in class_mapping:
                predicted_label = class_mapping[predicted_class]
            else:
                if last_valid_prediction is not None:
                    predicted_label = last_valid_prediction  # Use the last valid prediction if available
                else:
                    predicted_label = "WAKE"  # Default to "WAKE" for the very first prediction, if needed
            
            predictions.append(predicted_label)
            last_valid_prediction = predicted_label  # Update the last valid prediction
    
    return predictions

# V2
# import torch

# # Define the class mapping
# class_mapping = {
#     0: "W",  # Wake
#     1: "NR",  # Non-REM
#     2: "R"   # REM
# }

# # Define valid state transitions
# valid_transitions = {
#     "W": ["NR"],       # Wake can only transition to Non-REM
#     "NR": ["W", "R"],  # Non-REM can transition to Wake or REM
#     "R": ["W"]         # REM can only transition to Wake
# }

# def predict_sleep_stages(spectrograms, model):
#     """
#     Predict sleep stages with physiologically valid transitions.

#     Args:
#         spectrograms (torch.Tensor): Input data of shape (num_epochs, input_dim).
#         model (torch.nn.Module): Trained sleep stage prediction model.

#     Returns:
#         List[str]: Predicted sleep stages with valid transitions enforced.
#     """
#     model.eval()  # Set model to evaluation mode
#     predictions = []
    
#     last_valid_prediction = None  # Keep track of the last valid prediction
    
#     with torch.no_grad():  # Disable gradient calculation
#         num_datapoints = spectrograms.shape[0]  # Number of epochs
#         for i in range(num_datapoints):
#             epoch_spectrogram = torch.tensor(spectrograms[i]).unsqueeze(0)  # Add batch dimension
#             output = model(epoch_spectrogram)
            
#             # Get the predicted class (assuming output is logits, apply softmax if necessary)
#             predicted_class = output.argmax(dim=1).item()
#             predicted_label = class_mapping.get(predicted_class, None)
            
#             if predicted_label is None:
#                 # Handle invalid predictions gracefully
#                 predicted_label = last_valid_prediction or "W"  # Default to "W" if no valid history
#             elif last_valid_prediction is not None:
#                 # Validate the predicted transition
#                 if predicted_label not in valid_transitions[last_valid_prediction]:
#                     predicted_label = last_valid_prediction  # Revert to the last valid state
            
#             predictions.append(predicted_label)
#             last_valid_prediction = predicted_label  # Update the last valid prediction
    
#     return predictions
