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
def plot(examples,model):
    num = len(examples)

    num_rows = int(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig = plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    fig.canvas.manager.set_window_title("Prediction Results") 
    for i in range(num):
        image, true_label = examples[i]
        output, prediction = predict_digit(model,image)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis("off")
        plt.title(f"Prediction: {prediction}")
    plt.tight_layout()
    plt.show()

# Tests a pre-trained model on the first n examples, prints the results, and visualizes the predictions
def main():
    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

    # Load the first num_examples images from test set
    num_examples = 10
    test_examples = [(test_dataset[i][0], test_dataset[i][1]) for i in range (num_examples)]

    # Load the model (hard-coded below)
    model = load_model('trained_models/mnist_trained_model.pth')

    print("Testing on the first 10 example images:")
    for i, (image, label) in enumerate(test_examples):
        output, prediction = predict_digit(model,image)

        # Print the output values
        output_string = ', '.join([f'{val:.2f}' for val in output[0].tolist()])
        print(f"\nExample {i+1}:")
        print(f"Output values: {output_string}")
        print(f"Predicted digit: {prediction}")
        print(f"Correct Label: {label}")

    # Plot the images and predictions
    plot(test_examples, model)

if __name__ == "__main__":
    main()