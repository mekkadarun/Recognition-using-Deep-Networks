# CS5330-proj5
## 👥 Team Members

1. **Yuyang Tian**
2. **Arun Mekkad**

## 💻 Environment

- **🖥️ Yuyang Tian**: macOS 10.13.1 + CLion + CMake
- **🐧 Arun Mekkad**: Ubuntu 22.04 LTS + VS Code + CMake

### 📂 File Structure
    ```
    Proj5/
    ├──data/                  # 🖼️ MNIST data files
    ├── src/                  # 📁 Source files
        ├── test.py
        ├── train.py 
    ├── trained_models        # 📁 Directory for saving trained models
    ├── README.md             # 📖 Project documentation
    ```

## 📌 Tasks

-------------------------------------------------------------------------------------------------------------------
RUN train.py 

### TaskA 

The MNIST dataset is loaded using `load_mnist_test_data`, and the first 6 digits are visualized by calling `plot_mnist_samples`. Both steps are executed in the `main` function.

### TaskB

The CNN architecture is implemented in the `MyCNNNetwork` class. All layers are defined in the `__init__` method, and the `forward` method specifies how the input flows through those layers.

### TaskC

Trains a deep learning model on MNIST digits dataset. The model will be trained for 5 epochs, with each batch of training data containing 64 samples (batch_size = 64)

### TaskD

Saves the trained model in trained_models folder. Create this folder before running the code.

-------------------------------------------------------------------------------------------------------------------
RUN test.py

### TaskE

Loads the pre-trained model from local path and tests the model using example dataset.