'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 27th 2025
 * Purpose: Live testing for detection of handwritten digits
'''

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from test import load_model, predict_digit, ToTensor # Import ToTensor

import cv2
import numpy as np

# Preprocesses a PIL Image for digit recognition.
def preprocessimages(pil_image):
    # Convert PIL Image to NumPy array
    img = np.array(pil_image)
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate mean intensity to determine if inversion is required
    mean_intensity = np.mean(img)
    # If pixel intensity more than 127, invert the image
    if mean_intensity > 127:
        img = cv2.bitwise_not(img)
    # Define a kernel for dilation
    kernel = np.ones((5, 5), np.uint8)
    # Apply dilation using morphologyEx
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=3) # Adjusted kernel and iterations
    img = Image.fromarray(img)
    # Resize the image
    img = img.resize((28,28))
    # Convert image to tensor
    img_tensor = ToTensor()(img)
    return img_tensor

class RecognitionApp:
    # Initializes the main application window and its components.
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        self.canvas_width = 640
        self.canvas_height = 640
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.digit_label = tk.Label(root, text="Predicted Digit: ")
        self.digit_label.pack()

        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Starts drawing on the canvas when the left mouse button is pressed.
        self.canvas.bind("<Button-1>", self.start_draw)
        # Draws on the canvas as the mouse is moved while the left button is held down.
        self.canvas.bind("<B1-Motion>", self.draw)
        # Stops drawing when the left mouse button is released.
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.recognize_button = tk.Button(root, text="Recognize", command=self.recognize)
        self.recognize_button.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw_object = ImageDraw.Draw(self.image)

        # Loads the pre-trained digit recognition model.
        self.model = load_model('../trained_models/mnist_trained_model.pth')
        self.digit_labels = [0,1,2,3,4,5,6,7,8,9] # Digit labels

    # Sets the drawing flag to True and records the starting position.
    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    # Draws a line on the canvas and the PIL Image.
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=20, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            self.draw_object.line((self.last_x, self.last_y, x, y), fill="black", width=20)
            self.last_x, self.last_y = x, y

    # Sets the drawing flag to False and resets the last position.
    def stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    # Recognizes the drawn digit using the loaded model.
    def recognize(self):
        if self.model is None:
            return

        # Resize the PIL image to match canvas size before preprocessing
        resized_image = self.image.resize((self.canvas_width, self.canvas_height))

        # Preprocess the image using the function defined in this script
        img_tensor = preprocessimages(resized_image)

        if img_tensor is not None:
            # Recognize digit using the predict_digit function from test.py
            output, predicted_index = predict_digit(self.model, img_tensor)
            predicted_digit = self.digit_labels[predicted_index]
            self.digit_label.config(text=f"Predicted Digit: {predicted_digit}")

    # Clears the drawing canvas and resets the predicted digit label.
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw_object = ImageDraw.Draw(self.image)
        self.digit_label.config(text="Predicted Digit: ")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    clear_button = tk.Button(root, text="Clear", command=app.clear_canvas)
    clear_button.pack()
    root.mainloop()