'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 22nd 2025
 * Purpose: Loads a fine-tuned MNIST-based model trained on Greek letters (alpha, beta, gamma),
            evaluates it on a directory of handwritten Greek test images, and visualizes predictions.
            Includes helper functions for label mapping, transformation, model loading, prediction,
            and batch evaluation with accuracy reporting.
'''

import os
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from train import MyCNNNetwork
from train_greek import GreekTransform
from test import plot
from torchvision import datasets


# Returns forward and reverse mappings between class names and integer labels
def get_label_map():
    label_map = {'alpha': 0, 'beta': 1, 'gamma': 2, 'lambda':3, 'theta': 4}
    reverse_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_map


# Defines the transformation pipeline for test images: crop to 128x128, scale, crop, invert, normalize
def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((128, 128)),
        GreekTransform(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])


# Loads the fine-tuned Greek letter classification model from file
def load_model(path, num_classes):
    model = MyCNNNetwork()
    model.fc_layers[-1] = nn.Linear(
        in_features=model.fc_layers[-1].in_features,
        out_features=num_classes  # Match fine-tuned model (alpha, beta, gamma)
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Predicts the label of a single image tensor using the trained model
def predict_letter(model, image, reverse_map):
    model.eval()
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
        output = model(image)
        probabilities = torch.exp(output)
        _, predicted = torch.max(probabilities, 1)
        return output, reverse_map[predicted.item()]  # Return class name


# Evaluates all images in a directory (with subfolders per class), prints results and accuracy
def test_img_dir(model, img_dir, transform, reverse_map):
    correct = 0
    total = 0
    handwritten_examples = []

    for label_folder in os.listdir(img_dir):
        folder_path = os.path.join(img_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            image_tensor = transform(image)
            _, predicted_label = predict_letter(model, image_tensor, reverse_map)

            is_correct = predicted_label == label_folder
            result_text = "correct" if is_correct else "wrong"
            print(f"Image: {filename}, True: {label_folder}, Predicted: {predicted_label}, {result_text}")
            correct += int(is_correct)
            total += 1
            handwritten_examples.append((image_tensor, predicted_label))

    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%")
    plot(handwritten_examples)  # Visualize results


# Entry point: loads model and test set, runs evaluation
def main():
    # 1. Check how many classes are in the dataset
    train_path = '../data/greek_train'
    test_greek_path = '../data/greek_test'
    dataset = datasets.ImageFolder(root=train_path)
    letters = dataset.classes
    num_classes = len(letters)
    print(dataset.classes)
    print(f"Detected {num_classes} classes in '{train_path}': {letters}")
    # 2. find fine-tuned model path
    if num_classes == 3:
        model_path = '../trained_models/greek_trained_model.pth'
    else:
        model_path = f'../trained_models/greek_trained_{num_classes}.pth'

    # 3. Load models
    model = load_model(model_path, num_classes)

    # 4. Testing
    transform = get_transform()
    _, reverse_map = get_label_map()
    test_img_dir(model, test_greek_path, transform, reverse_map)


if __name__ == "__main__":
    main()