'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 20th 2025
 * Purpose: Define and train a CNN model on the MNIST handwritten digit dataset, 
            evaluate its performance during training, visualize the training progress, 
            and save the trained model to disk.
'''
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")  # Standard backend
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
from util import load_mnist_data

# Custom CNN
class MyCNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_percent = 0.3
        # CNN related layer, including conv, ReLU, pooling and dropout
        self.cnn_layers = nn.Sequential(
            # First Conv layer: C_out = 10, kernel size = 5
            nn.Conv2d(1, 10, 5),
            # pooling layer: Max pool with a 2x2 window
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second Conv layer
            nn.Conv2d(10, 20,5),
            nn.ReLU(),
            nn.Dropout(self.drop_percent),
            nn.MaxPool2d(2),
        )
        # Calculating the size after convolutions and pooling
        # Input: 28x28 -> Conv1(5x5) -> 24x24 -> MaxPool(2x2) -> 12x12
        # -> Conv2(5x5) -> 8x8 -> MaxPool(2x2) -> 4x4
        # So the feature map size is 4x4 with 20 channels = 20*4*4 = 320
        self.fc_layers = nn.Sequential(
            # Linear layer takes (20x
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )


    # Computes a forward pass for the network
    def forward(self, x):
        # input (x): [batch_size, 1, 28, 28]
        # output: [batch_size, 1]

        # Extract features by Conv layers
        x = self.cnn_layers(x)
        # Flatten the output for Fully connected layers
        x = x.flatten(1)
        # Pass through fully connected layers
        x = self.fc_layers(x)
        # Apply log softmax on the output
        return F.log_softmax(x, 1)

# Display the first six sample digits
def plot_mnist_samples(data):
    '''
    :param data: the MNIST dataset
    '''
    figure = plt.figure(figsize=(8, 6))
    rows, cols = 2, 3
    for i in range(1, cols * rows + 1):
        img, label = data[i]
        ax = figure.add_subplot(rows, cols, i)
        ax.set_title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.tight_layout()
    plt.show()

# Function to evaluate the model on a given dataset
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Function that trains the neural network for a specified number of epochs
def train_network(model, test_loader, train_loader, criterion, optimizer, num_epochs = 5):
    
    # Initialize train and test loss lists
    train_losses =[]
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start = timeit.default_timer()
    # Set the model to training mode
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
    
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        
        # Training set evaluation
        train_avg_loss, train_accuracy = evaluate_model(model, train_loader,criterion)
        train_losses.append(train_avg_loss)
        train_accuracies.append(train_accuracy)
        
        # Test set evaluation
        test_avg_loss, test_accuracy = evaluate_model(model, test_loader,criterion)
        test_losses.append(test_avg_loss)
        test_accuracies.append(test_accuracy)

        print(f"EPOCH [{epoch+1}/{num_epochs}],"
              f"Train Loss: {train_avg_loss:.4f}, Train accuracy: {train_accuracy:.2f}%"
              f"Test Loss: {test_avg_loss : .4f}, Test accuracy: {test_accuracy:.2f}%")
        
    end = timeit.default_timer()
    print(f"Total training time: {end - start:.2f}s")
    return train_losses, test_losses, train_accuracies, test_accuracies

# Function to plot training and testing error
def plot_error(train_losses, test_losses):

    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(10,6))
    fig.canvas.manager.set_window_title("Errors Over Epochs Plot") 
    plt.plot(epochs, train_losses, 'b-o', label = 'Training error (loss)')
    plt.plot(epochs, test_losses, 'r-o', label = 'Testing error (loss)')
    plt.title('Training and Testing Errors over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot training and testing accuracies
def plot_accuracy(train_accuracies, test_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    fig = plt.figure(figsize=(10,6))
    fig.canvas.manager.set_window_title("Accuracy Over Epochs Plot") 
    plt.plot(epochs, train_accuracies, 'm-o', label = 'Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g-o', label = 'Testing Accuracy')
    plt.title('Training and Testing Accuracies over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Trains a new model using MNIST digit dataset and save the model to trained_models directory
def main(argv):
    # Load MNIST data
    train_data, test_data = load_mnist_data()

    # Plot taskA
    plot_mnist_samples(test_data)

    # Choose a batch size
    batch_size = 256

    # Create data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle= True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle= False)

    # Instantiate the network
    model = MyCNNNetwork()

    # Define the loss function
    criterion = nn.NLLLoss()

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train the network
    num_epochs = 5
    train_losses, test_losses, train_accuracies, test_accuracies = train_network(model=model, test_loader=test_loader, train_loader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

    # Plot error and accuracy over epochs
    plot_error(train_losses, test_losses)
    plot_accuracy(train_accuracies, test_accuracies)

    # Save the trained model
    torch.save(model.state_dict(), '../trained_models/mnist_trained_model.pth')
    print(f"Trained model saved to mnist_trained_model.pth. Check trained_models folder")

    return

if __name__ == "__main__":
    main(sys.argv)

