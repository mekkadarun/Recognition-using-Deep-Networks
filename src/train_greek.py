'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 25th 2025
 * Purpose: Fine-tunes a pre-trained MNIST digit recognition model to classify
            three Greek letters (alpha, beta, gamma) using transfer learning.
            loads the original model, freezes the base layers, replaces
            the final classification layer, trains on the Greek dataset, and saves
            the resulting model. It also includes utilities for preprocessing,
            prediction, and visualizing training loss over epochs.
'''

import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from torchvision import datasets
from train import MyCNNNetwork, evaluate_model

# Custom transformation to mimic MNIST-style preprocessing for Greek letter images
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)  # Convert RGB to grayscale
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)  # Scale to ~36x36
        x = torchvision.transforms.functional.center_crop(x, (28, 28))  # Center crop to MNIST size
        return torchvision.transforms.functional.invert(x)  # Invert to white-on-black like MNIST


# Returns forward and reverse label mappings (e.g., 'alpha' ↔ 0)
def get_label_map():
    label_map = {'alpha': 0, 'beta': 1, 'gamma': 2}
    reverse_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_map


# Returns the full preprocessing pipeline used for both training and inference
def get_transform():
    return torchvision.transforms.Compose([
        GreekTransform(),  # Affine scaling, crop, invert
        torchvision.transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean/std
    ])


# Loads a pretrained model from a file (assumes correct architecture)
def load_model(path):
    model = MyCNNNetwork()
    model.load_state_dict(torch.load(path))
    return model


# Applies transfer learning: freezes base network and replaces the final layer
def transfer_learning(model):
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Replace last layer to output 3 Greek letter classes
    model.fc_layers[-1] = nn.Linear(
        in_features=model.fc_layers[-1].in_features,
        out_features=3
    )

    print("\nModel Architecture after transfer learning:")
    print(model)
    return model


# Trains the model for a given number of epochs and returns training loss/accuracy per epoch
def train_network(model, train_loader, criterion, optimizer, num_epochs=5):
    train_losses = []
    train_accuracies = []

    start = timeit.default_timer()
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss, accuracy = evaluate_model(model, train_loader, criterion)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"EPOCH [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    end = timeit.default_timer()
    print(f"Total training time: {end - start:.2f}s")
    return train_losses, train_accuracies


# Predicts the Greek letter label for a single input image tensor
def predict_letter(model, image, reverse_map):
    model.eval()
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        output = model(image)
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities
        _, predicted = torch.max(probabilities, 1)
        return output, reverse_map[predicted.item()]  # Return label name


# Plots the training loss over epochs
def plot_error(train_losses):
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Errors Over Epochs Plot")
    plt.plot(epochs, train_losses, 'b-o', label='Training error (loss)')
    plt.title('Training Errors over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


# Main pipeline: load pretrained model, apply transfer learning, train on Greek letters, save model
def main():
    model_path = '../trained_models/mnist_trained_model.pth'
    train_path = '../data/greek_train'

    pre_model = load_model(model_path)  # Load original MNIST model
    model = transfer_learning(pre_model)  # Adapt it to Greek classification

    transform = get_transform()
    greek_train_loader = DataLoader(
        datasets.ImageFolder(root=train_path, transform=transform),
        batch_size=5,
        shuffle=True
    )

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    train_losses, _ = train_network(model, greek_train_loader, criterion, optimizer, num_epochs=25)

    # Plot loss curve to tune nepoch visually
    plot_error(train_losses)

    # Save fine-tuned Greek model
    torch.save(model.state_dict(), '../trained_models/greek_trained_model.pth')
    print(f"Trained model saved to greek_trained_model.pth. Check trained_models folder")


if __name__ == "__main__":
    main()
