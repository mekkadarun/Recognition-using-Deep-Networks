# CS5330-proj5
## 👥 Team Members

1. **Yuyang Tian**
2. **Arun Mekkad**

## 💻 Environment

- **🖥️ Yuyang Tian**: macOS 10.13.1 + CLion + CMake
- **🐧 Arun Mekkad**: Ubuntu 22.04 LTS + VS Code + CMake

## 📌 Tasks

### TaskA 

The MNIST dataset is loaded using `load_mnist_test_data`, and the first 6 digits are visualized by calling `plot_mnist_samples`. Both steps are executed in the `main` function.

### TaskB

The CNN architecture is implemented in the `MyCNNNetwork` class. All layers are defined in the `__init__` method, and the `forward` method specifies how the input flows through those layers.

### TaskC

Trains a deep learning model on MNIST digits dataset. The model will be trained for 5 epochs, with each batch of training data containing 64 samples (batch_size = 64)

### TaskD

Saves the trained model in trained_models folder. Create this folder before running the code.