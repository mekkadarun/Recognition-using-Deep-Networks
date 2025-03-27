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
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from train import MyCNNNetwork, plot_error, plot_accuracy, train_network

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


# Applies transfer learning: freezes base network and replaces the final layer of the MNIST network to match 'num_classes' (frozen base).
def transfer_learning(model, num_classes):
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Replace last layer to output 3 Greek letter classes
    model.fc_layers[-1] = nn.Linear(
        in_features=model.fc_layers[-1].in_features,
        out_features=num_classes
    )

    print(f"\nModel Architecture after transfer learning for {num_classes} classes:")
    print(model)
    return model


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


# Main pipeline: load pretrained model, apply transfer learning, train on Greek letters, save model
def main(argv):
    model_path = '../trained_models/mnist_trained_model.pth'
    if len(argv) > 1 and argv[1] == '--extension':
        train_path = '../data/greek_train_5'
        test_path = '../data/greek_test_5'
    else:
        train_path = '../data/greek_train_3'
        test_path = '../data/greek_test_3'

    # 1. Load the original MNIST base model
    pre_model = load_model(model_path)

    # 2. Check how many classes are in the dataset
    #    For example, alpha/beta/gamma => 3 classes
    dataset = datasets.ImageFolder(root=train_path)
    letters = dataset.classes
    num_classes = len(letters)
    print(dataset.classes)
    print(f"Detected {num_classes} classes in '{train_path}': {letters}")

    # 3. Adapt model to have 'num_classes' outputs
    model = transfer_learning(pre_model, num_classes)

    # 4. Create data loader with GreekTransform
    transform = get_transform()
    greek_train_loader = DataLoader(
        datasets.ImageFolder(root=train_path, transform=transform),
        batch_size=5,
        shuffle=True
    )
    greek_test_loader = DataLoader(
        datasets.ImageFolder(root=test_path, transform=transform),
        batch_size=5,
        shuffle=True
    )

    # 5. Train the adapted model
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    train_losses, test_losses, train_accuracies, test_accuracies = train_network(model, greek_test_loader, greek_train_loader, criterion, optimizer, num_epochs=50)

    # 6. Plot results
    plot_error(train_losses, test_losses)
    plot_accuracy(train_accuracies, test_accuracies)

    # 7. Save fine-tuned model
    if num_classes == 3:
        out_path = '../trained_models/greek_trained_model.pth'
    else:
        out_path = f'../trained_models/greek_trained_{num_classes}.pth'

    torch.save(model.state_dict(), out_path)
    print(f"Trained model saved to {out_path} (check trained_models folder)")

if __name__ == "__main__":
    main(sys.argv)
