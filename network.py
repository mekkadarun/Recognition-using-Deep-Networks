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

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        pass
    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        return x
# useful functions with a comment for each function
def load_mnist_test_data():
    '''
    Downloads and loads the MNIST dataset.
    :return: MNIST test dataset
    '''
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())
    return mnist_test
def plot_mnist_samples(data):
    '''
    Display the first six sample digits
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

