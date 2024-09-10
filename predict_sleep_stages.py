# import torch

# class_mapping = {
#     0: "WAKE",
#     1: "NREM",
#     2: "REM",
#     3: "Artifact"  # This is for error/unknown cases
# }

# def predict_sleep_stages(spectrograms, model):
#     model.eval()  # Set model to evaluation mode
#     predictions = []
    
#     with torch.no_grad():  # Disable gradient calculation
#         num_datapoint = spectrograms.shape[0]  # Number of epochs
#         for i in range(num_datapoint):
#             epoch_spectrogram = torch.tensor(spectrograms[i]).unsqueeze(0)  # Add batch dimension
#             output = model(epoch_spectrogram)
            
#             # Get the predicted class (assuming output is logits, apply softmax if necessary)
#             # TODO: Look into prediction with +1
#             predicted_class = output.argmax(dim=1).item()
#             predicted_label = class_mapping.get(predicted_class , "Artifact")  # +1 to match class_mapping
#             predictions.append(predicted_label)
    
#     return predictions

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
