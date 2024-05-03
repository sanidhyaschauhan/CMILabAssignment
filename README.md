# Image Classification of Sclerotic and Non-Sclerotic Glomeruli

## Project Overview
This repository is dedicated to the development and evaluation of machine learning models for the binary classification of sclerotic and non-sclerotic glomeruli using microscopic images. This complex task is tackled using three different modeling approaches: logistic regression, a custom CNN, and a pretrained ResNet-50. The project not only aims to achieve high accuracy in classification but also to explore the strengths and limitations of each modeling approach in handling medical image data.

## Models Overview

Here is a quick summary of the models and their performances:

| Model               | Test Set Accuracy |
|---------------------|-------------------|
| Logistic Regression | 90.5%             |
| Simple CNN          | 98.3%             |
| ResNet-50           | 99%               |

## Files in the Repository

- **`LinearRegression.ipynb`**: Explores the use of logistic regression for initial model benchmarking.
- **`CNNClassifier.ipynb`**: Develops a custom CNN tailored to the specific needs of glomeruli image classification.
- **`resnet50classifier.ipynb`**: Adapts a pretrained ResNet-50 model to the task, leveraging deep learning advancements for improved accuracy.
- **`evaluation.py`**: Provides utilities for model evaluation and metric calculation.

## Detailed Workflow and Model Insights

### Step 1: Logistic Regression (LinearRegression.ipynb)

#### Purpose
To establish a baseline for classification performance using a simple logistic regression model, which helps identify key features and dataset characteristics.

#### Challenges
- **Image Size Variability**: The initial dataset contained images of varying sizes, which complicates the creation of a uniform feature set for logistic regression.
- **Class Imbalance**: A significant imbalance between the classes which can bias the model toward the majority class.

#### Solutions
- **Image Preprocessing**: Implemented image resizing and padding to standardize the input size for all images.
- **Rebalancing Techniques**: Applied data augmentation techniques such as image rotations and flips to enhance the dataset's diversity and balance.

#### Results
The logistic regression model achieved a respectable accuracy of 90.5%, providing valuable insights into the data's characteristics and the feasibility of using simpler models for preliminary analysis.

### Step 2: Custom CNN Model (CNNClassifier.ipynb)

#### Purpose
To significantly enhance model performance by leveraging a CNN's capability to capture spatial hierarchies in image data.

#### Model Architecture
- **Layers**: Consists of several convolutional layers, each followed by max pooling. Batch normalization is included to maintain stability and speed up the network's training. The network ends with dense layers for classification.

#### Challenges
- **Computational Resources**: Initially faced issues with large image sizes overwhelming the available computational resources.

#### Solutions
- **Optimized Image Size**: Reduced the image dimensions to 128x128 pixels to accommodate the hardware limitations without substantially sacrificing model accuracy.

#### Results
This tailored CNN architecture improved accuracy to 98.3%, validating the effectiveness of convolutional networks in handling image classification tasks, especially with well-preprocessed input data.

### Step 3: Using Pretrained Model - ResNet-50 (resnet50classifier.ipynb)

#### Purpose
To utilize the powerful, pretrained ResNet-50 model to push the boundaries of accuracy and performance in our classification task.

#### Model Adaptation
- **Fine-tuning**: Modified the last few layers of the ResNet-50 to better suit our binary classification needs.

#### Challenges
- **Overfitting**: Managing the model's complexity to prevent overfitting while maintaining high accuracy on unseen data.

#### Solutions
- **Data Augmentation**: Enhanced the dataset with more diverse image transformations to improve the model's ability to generalize.

#### Results
Achieved an exceptional accuracy of 99%, demonstrating the potential of using advanced pretrained models in specialized areas such as medical image analysis.

### Evaluation and Metrics (evaluation.py)

#### Functionality
This script is crucial for assessing the performance of different models. It processes test images, loads the trained models, and predicts outcomes, providing a standardized evaluation framework.

#### Usage
```bash
python evaluation.py --test_data_path [path_to_test_data] --model_path [path_to_saved_model]
