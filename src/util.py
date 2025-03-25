'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 25th 2025
 * Purpose: Util functions for loading data and models
'''

from train import MyCNNNetwork # Import network definition from train.py
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

# Downloads and loads a train and test data from MNIST dataset
def load_mnist_data():
    '''
    Load train and test data
    :return: Train and test data
    '''
    train_data = datasets.MNIST(root="../data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="../data", train=False, download=True, transform=ToTensor())

    return train_data, test_data

# Loads the trained model from path
def load_model(path):
    # Create an instance of MyCNNNetwork
    model = MyCNNNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

