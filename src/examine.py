'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 24th 2025
 * Purpose: Examine a pre-trained MNIST model, analyzes and visualizes the filters from specified layer,
            and shows the result of applying these filters to the first MNIST training image.
'''

import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from util import load_model
from train import MyCNNNetwork
from torchvision.transforms import ToTensor

# Analyze the specified layer
def analyze_layer(model: MyCNNNetwork, layer: int):
    # Print the model architecture
    print("\nModel Architecture:")
    print(model)

    # Analyze the first convolutional layer
    conv_layer = model.cnn_layers[layer]
    weights = conv_layer.weight.data

    # Print the shape of the weights
    print(f"\nShape of the layer {layer} weights:",weights.shape)
    
    # Print the filter weights
    print(f"\nWeights of the layer {layer} filter:")
    print(weights[0,0])

    return weights

# Visualize the filters
def visualize_filters(weights, first_image = None):
    fig = plt.figure(figsize=(8,6))
    fig.canvas.manager.set_window_title("Visualization of Convolutional Layer")

    num_filters = weights.shape[0]
    num_rows = 3
    num_cols = 4

    for i in range(num_filters):
        plt.subplot(num_rows,num_cols, i+1)
        filter_weight = weights[i,0].numpy()
        plt.imshow(filter_weight, cmap='gray')
        plt.title(f"Filter {i+1}")
        plt.xticks()
        plt.yticks()
    plt.tight_layout()
    plt.show()

# Visualize the filters and filtered images side-by-side
def visualize_filtered_images(weights, first_image):
    fig = plt.figure(figsize=(12,15))
    fig.canvas.manager.set_window_title("Visualization of filtered images")
    
    num_filters = weights.shape[0]
    num_rows = 5
    num_cols = 4

    # Visualize the filters and images
    with torch.no_grad():
        for i in range(num_filters):
            filter_weight = weights[i,0].numpy()
            filtered_image = cv2.filter2D(first_image, -1, filter_weight)
            
            # Plot the filter
            plt.subplot(num_rows, num_cols, 2 * i+1)
            plt.imshow(filter_weight,cmap='gray')
            plt.title(f"Filter {i+1}")
            plt.axis('off')

            # Plot the filtered image
            plt.subplot(num_rows, num_cols, 2 * i+2)
            plt.imshow(filtered_image,cmap='gray')
            plt.title(f"Filtered Image {i+1}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# Loads the model and train dataset from MNIST, visualizes the filters from specified convolutional layer
def main():
    model_path = '../trained_models/mnist_trained_model.pth'
    # Load the model
    model  = load_model(model_path)

    # Load train dataset
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True,transform=ToTensor())

    # Analyze layer 0 and get the weights
    weights = analyze_layer(model, 0)

    # Visualize the filters
    visualize_filters(weights)

    # Get the first image
    first_image_tensor, _ = train_dataset[0]
    first_image = first_image_tensor.squeeze().numpy()

    # Visualize the filters and images
    visualize_filtered_images(weights, first_image)

if __name__ == "__main__":
    main()