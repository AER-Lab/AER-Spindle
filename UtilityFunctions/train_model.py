from .utils import classification_accuracy
import time
def train_model(model, criterion, optimizer, train_loader, epochs=10, progress=None, label=None, time_label=None, start_time=None):
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

         # Calculate elapsed time and estimate time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        estimated_time_remaining = avg_time_per_epoch * remaining_epochs

         # Format estimated time remaining for display
        if estimated_time_remaining < 60:
            time_remaining_text = f"{estimated_time_remaining:.2f} seconds"
        elif estimated_time_remaining < 3600:
            time_remaining_text = f"{estimated_time_remaining / 60:.2f} minutes"
        else:
            time_remaining_text = f"{estimated_time_remaining / 3600:.2f} hours"

        time_label.config(text=f"Estimated time remaining: {time_remaining_text}")
        progress.update()

    print("Training complete!")
    return model
