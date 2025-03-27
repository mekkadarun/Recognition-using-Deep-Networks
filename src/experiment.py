'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 25th 2025
 * Purpose: Experiment with different parameters
'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
import csv
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Defines a configurable Convolutional Neural Network (CNN) model.
class ConfigCnn(nn.Module):
    # Initializes the CNN with specified number of layers, filter sizes, filters per layer, and dropout rate.
    def __init__(self, num_layers, filter_sizes, num_filters_per_layer, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.filter_sizes = filter_sizes
        self.num_filters_per_layer = num_filters_per_layer
        self.dropout_rate = dropout_rate

        self.cnn_layers = nn.Sequential()
        channel_inputs = 1

        for i in range(num_layers):
            kernel_size = filter_sizes[i]
            channel_outputs = num_filters_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(channel_inputs, channel_outputs, kernel_size))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            channel_inputs = channel_outputs

        # Calculates the output size after the convolutional layers.
        def get_output_size(input_size, num_layers, kernel_sizes):
            size = input_size
            for k in kernel_sizes:
                size = size - k + 1
                size = size // 2
            return size

        conv_output_size = get_output_size(28, num_layers, filter_sizes)
        if num_layers > 0:
            fc_input_size = num_filters_per_layer[-1] * conv_output_size * conv_output_size
        else:
            fc_input_size = 28 * 28

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    # Defines the forward pass of the CNN model.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return F.log_softmax(x, 1)

# Loads the MNIST dataset for training and testing.
def load_mnist_data():
    train_data = datasets.FashionMNIST(root="../data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="../data", train=False, download=True, transform=ToTensor())
    return train_data, test_data

# Evaluates the performance of the model on a given data loader.
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Trains the neural network model.
def train_network(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    start_time = timeit.default_timer()
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    return test_accuracy, training_time

# Main function to run experiments with different hyperparameters.
def main(argv):
    num_conv_layers_options = [1, 2, 3]
    filter_size_options_base = [3, 5]
    num_filters_per_layer_options_base = [16, 32, 64]
    dropout_rate_options = [0.0, 0.25, 0.5]  # Adding dropout rate options
    num_epochs_options = [10, 20, 30] # Trying three epoch values

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, test_data = load_mnist_data()
    batch_size = 256
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    criterion = nn.NLLLoss()
    learning_rate = 0.001

    results = []

    for num_layers in num_conv_layers_options:
        for filter_size in filter_size_options_base:
            for num_filters_per_layer in num_filters_per_layer_options_base:
                for dropout_rate in dropout_rate_options:  # Looping through dropout rates
                    for num_epochs in num_epochs_options: # Looping through epoch values
                        print(f"\n--- Experimenting with: ---")
                        print(f"  Num Conv Layers: {num_layers}")
                        print(f"  Filter Size: {filter_size}x{filter_size}")
                        print(f"  Num Filters per Layer: {num_filters_per_layer}")
                        print(f"  Dropout Rate: {dropout_rate}")
                        print(f"  Num Epochs: {num_epochs}")

                        filter_sizes = [filter_size] * num_layers
                        num_filters = [num_filters_per_layer] * num_layers

                        # More robust check for filter size based on the number of layers
                        if num_layers == 3 and filter_size > 3:
                            print("  Skipping this configuration as filter size is too large for 3 layers.")
                            continue
                        if num_layers == 2 and filter_size > 5:
                            print("  Skipping this configuration as filter size might be too large for 2 layers.")
                            continue

                        model = ConfigCnn(num_layers, filter_sizes, num_filters, dropout_rate).to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

                        test_accuracy, training_time = train_network(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

                        print(f"  Test Accuracy: {test_accuracy:.2f}%")
                        print(f"  Training Time: {training_time:.2f} seconds")

                        results.append({
                            'num_conv_layers': num_layers,
                            'filter_size': filter_size,
                            'num_filters_per_layer': num_filters_per_layer,
                            'dropout_rate': dropout_rate,
                            'num_epochs': num_epochs,
                            'test_accuracy': test_accuracy,
                            'training_time': training_time
                        })

    # Save the results to a CSV file
    csv_file = "experiment_results_basic.csv"
    fieldnames = results[0].keys() if results else []

    if fieldnames:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_file}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main(sys.argv)