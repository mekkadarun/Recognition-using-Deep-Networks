# Author: Yuyang Tian, Arun Mekkad
# Project 5
import sys
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use("TkAgg")  # Standard backend
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Custom CNN
class MyCNNNetwork(nn.Module):
    def __init__(self):
        super.__init__(MyCNNNetwork, self).__init__()
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


    # computes a forward pass for the network
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
# useful functions with a comment for each function
# Downloads and loads the MNIST dataset.
def load_mnist_test_data():
    '''
    :return: MNIST test dataset
    '''
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())
    return mnist_test
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

def train_network( arguments ):
    return
# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # main function code
    test_data = load_mnist_test_data()
    plot_mnist_samples(test_data)
    return
if __name__ == "__main__":
    main(sys.argv)

