from utils import classification_accuracy

def train_model(model, criterion, optimizer, train_loader, epochs=10, progress=None, label=None):
    total_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0  # Track accuracy across batches

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)

            # Skip batch if output and label sizes don't match
            if outputs.size(0) != labels.size(0):
                continue

            # Compute loss and perform backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy for the current batch
            accuracy = classification_accuracy(outputs, labels)
            running_accuracy += accuracy.item()

            # Update progress bar for each batch (if progress bar is provided)
            if progress and label:
                current_progress = ((epoch * total_batches + batch_idx + 1) / (epochs * total_batches)) * 100
                progress['value'] = current_progress
                label.config(text=f"{current_progress:.2f}% completed")
                progress.update()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total_batches
        epoch_accuracy = running_accuracy / total_batches

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training complete!")
    return model