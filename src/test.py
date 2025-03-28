'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 22nd 2025
 * Purpose: Loads a pre-trained MNIST digit recognition model, 
            evaluates it on a specified number of test examples, 
            and visualizes the predictions.
'''
import torch
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from train import MyCNNNetwork # Import network definition from train.py
import math
import os
import sys
from PIL import Image
import cv2
import numpy as np


# Loads the trained model from path
def load_model(path):
    # Create an instance of MyCNNNetwork
    model = MyCNNNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Predicts the digit for a given image tensor
def predict_digit(model, image):
    with torch.no_grad():
        # Add a batch dimension
        image = image.unsqueeze(0)
        output = model(image)
        # Convert log-probabilities to probabilities 
        probabilities = torch.exp(output) 
        _,predicted = torch.max(probabilities,1)
    return output, predicted.item()

# Plot the example images with their predictions
def plot(examples):
    num = len(examples)

    num_rows = int(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig = plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    fig.canvas.manager.set_window_title("Prediction Results") 
    for i in range(num):
        image, prediction = examples[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis("off")
        plt.title(f"Prediction: {prediction}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Pre-process the images of handwritten examples
def preprocess_images(image_path):
    # Convert to grayscale
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate mean intensity to determine if inversion is required
    mean_intensity = np.mean(img)
    # If pixel intensity more than 255/2, invert the image
    if mean_intensity > 127:
        img = cv2.bitwise_not(img)
        print(f"INVERTED!")
    # Define a kernel for dilation
    kernel = np.ones((5, 5), np.uint8)
    # Apply dilation using morphologyEx
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, 10 * kernel,iterations=3)
    img = Image.fromarray(img)
    # Resize the image
    img = img.resize((28,28))
    # Convert image to tensor
    img_tensor = ToTensor()(img)
    return img_tensor

# Print the prediction results
def print_prediction_result(filename, output, predicted_digit):
        output_string = ', '.join([f'{val:.2f}' for val in output[0].tolist()])
        print(f"\nProcessed image: {filename}:")
        print(f"Output values: {output_string}")
        print(f"Predicted digit: {output_string}")
        print(f"Correct Label: {predicted_digit}")

# Runs the test on the first num_examples images from mnist dataset
def test_on_mnist(model, num_examples=10):
    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=ToTensor())
    test_examples = [(test_dataset[i][0], test_dataset[i][1]) for i in range (num_examples)]
    print(f"Testing on the first {num_examples} example images from MNIST:")
    results_for_plot =[]
    # Iterate through all the images
    for i, (image, label) in enumerate(test_examples):
        # Run the prediction method
        output, prediction = predict_digit(model, image)
        # Print the results
        print_prediction_result(filename=f"Example {i+1}", output=output, predicted_digit=prediction)
        # Append for plotting
        results_for_plot.append((image, prediction))
    # Plot the results
    plot(results_for_plot)

# Runs the test on the handwritten images using pre-trained model
def test_on_handwritten_images(model, img_dir):
    print(f"Pre-processing images from directory: {img_dir}")
    # Store all the images in a list
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir,f))]
    handwritten_examples = []
    # Iterate through all the images
    for i, filename in enumerate(img_files):
        # Find the path to each images
        img_path = os.path.join(img_dir, filename)
        # Pre-process the handwritten images
        processed_tensor = preprocess_images(img_path)
        if processed_tensor is not None:
            # Run the prediction test
            output, prediction = predict_digit(model, processed_tensor)
            # Print the results
            print_prediction_result(filename, output, prediction)
            # Append for plotting
            handwritten_examples.append((processed_tensor,prediction))
    # Plot the results
    plot(handwritten_examples)

# Tests a pre-trained model on either the first n images from MNIST dataset or provided directory path
def main(argv):
    # Load the model (hard-coded below)
    model = load_model('../trained_models/mnist_trained_model.pth')

    if len(argv) > 1:
        img_dir = argv[1]
        test_on_handwritten_images(model, img_dir)
    else:
        test_on_mnist(model)
    return

if __name__ == "__main__":
    main(sys.argv)