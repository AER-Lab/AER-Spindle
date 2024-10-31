from utils import classification_accuracy

def train_model(model, criterion, optimizer, train_loader, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.size(0) != labels.size(0):
                continue
            # Convert one-hot encoded labels to class indices
            # Ensure labels are 1D tensor of class indices
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #Implement the classification accuracy  which takes model predictions and ground truth data and tells us how well the model is preforming correct predictions/total predictions
            accuracy = classification_accuracy(outputs, labels)

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy.item()}")
    return model