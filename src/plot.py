'''
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: March 24th 2025
 * Purpose: Visualize the changes in parameters on test accuracy
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("experiment_results_basic.csv")

# Plot Test Accuracy vs. Number of Convolutional Layers
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='num_conv_layers', y='test_accuracy')
plt.title('Test Accuracy vs. Number of Convolutional Layers')
plt.xlabel('Number of Convolutional Layers')
plt.ylabel('Test Accuracy (%)')
plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test Accuracy vs. Filter Size
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='filter_size', y='test_accuracy')
plt.title('Test Accuracy vs. Filter Size')
plt.xlabel('Filter Size')
plt.ylabel('Test Accuracy (%)')
plt.xticks(ticks=[0, 1], labels=['3x3', '5x5'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test Accuracy vs. Number of Filters per Layer
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='num_filters_per_layer', y='test_accuracy')
plt.title('Test Accuracy vs. Number of Filters per Layer')
plt.xlabel('Number of Filters per Layer')
plt.ylabel('Test Accuracy (%)')
plt.xticks(ticks=[0, 1, 2], labels=['16', '32', '64'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test Accuracy vs. Dropout Rate
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='dropout_rate', y='test_accuracy')
plt.title('Test Accuracy vs. Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Test Accuracy (%)')
plt.xticks(ticks=[0, 1, 2], labels=['0.0', '0.25', '0.5'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test Accuracy vs. Number of Epochs
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='num_epochs', y='test_accuracy')
plt.title('Test Accuracy vs. Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Test Accuracy (%)')
plt.xticks(ticks=[0, 1, 2], labels=['10', '20', '30'])
plt.grid(True)
plt.tight_layout()
plt.show()