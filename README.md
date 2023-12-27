# Brain Tumor Detection using Convolutional Neural Networks (CNN)

## Overview

This Python script utilizes a Convolutional Neural Network (CNN) model to detect brain tumors from MRI images. The model is built using TensorFlow and Keras libraries, trained on a dataset containing brain MRI images classified into two categories: 'yes' (indicating the presence of a tumor) and 'no' (no tumor present).

## Dataset

The dataset used for training comprises MRI images categorized into 'yes' and 'no' folders, representing positive and negative cases of brain tumors, respectively. Each image is resized to a square shape of 128x128 pixels and converted to grayscale for processing.

## Model Architecture

The CNN model architecture includes:

- Convolutional layers with 32 and 64 filters, using a 3x3 kernel size and ReLU activation.
- MaxPooling layers for downsampling.
- Dense layers with ReLU activation and a final output layer using the sigmoid activation function for binary classification.

## Usage

### Dataset Preparation:
1. Organize MRI images into 'yes' and 'no' folders.
2. Ensure all images are of the same size (128x128 pixels).

### Running the Script:
- Update the `data_path` variable with the path to your dataset.
- Execute the Python script to preprocess data, build the CNN model, and train it on the dataset.

## Model Evaluation

The model is trained using 80% of the dataset for training and 20% for testing. Evaluation metrics include accuracy and loss, visualized through training and validation accuracy/loss plots.

## Dependencies

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn (for train-test split)

## Future Improvements

Potential enhancements for the model:

- Hyperparameter tuning for improved performance.
- Data augmentation techniques to further generalize the model.

## Credits

- **TensorFlow and Keras** - Deep learning framework and library.
- **OpenCV** - Image processing library in Python.
