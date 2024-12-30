# CovidCNN TORCH
# Chest X-ray Classification using CNN

This project aims to classify chest X-ray images into three categories: **COVID-19**, **Pneumonia**, and **Normal** using a Convolutional Neural Network (CNN) built with PyTorch.

## Dataset

The dataset is divided into two folders:
- **Training Set**: Contains images used for training the model.
- **Test Set**: Contains images used for evaluating the model's performance.

Each image is resized to 512x512 pixels, normalized, and converted into a tensor for input to the model.

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed to effectively extract features from X-ray images for classification. It consists of the following components:

### 1. **Convolutional Layers**:
   - **Layer 1**: 
     - Input: 3 channels (RGB image)
     - Output: 32 channels
     - Operation: 3x3 convolution with ReLU activation followed by 2x2 max pooling
     - Dropout with a rate of 0.2 to prevent overfitting
     
   - **Layer 2**: 
     - Input: 32 channels
     - Output: 64 channels
     - Operation: 3x3 convolution with ReLU activation followed by 2x2 max pooling
     - Dropout with a rate of 0.2
     
   - **Layer 3**: 
     - Input: 64 channels
     - Output: 128 channels
     - Operation: 3x3 convolution with ReLU activation followed by 2x2 max pooling
     - Dropout with a rate of 0.2
     
   - **Layer 4**: 
     - Input: 128 channels
     - Output: 256 channels
     - Operation: 3x3 convolution with ReLU activation followed by 2x2 max pooling
     - Dropout with a rate of 0.2
     
   - **Layer 5**: 
     - Input: 256 channels
     - Output: 512 channels
     - Operation: 3x3 convolution with ReLU activation followed by 2x2 max pooling
     - Dropout with a rate of 0.2
     
   These convolutional layers help the model learn hierarchical features from the input images, starting from simple edges and textures to more complex shapes and patterns.

### 2. **Fully Connected Layers**:
   - **Flattening**: After the convolutional layers, the output is flattened to a 1D vector to connect with the fully connected layers.
   - **Layer 1**: 
     - Input: 512 * 16 * 16 (flattened)
     - Output: 4096 neurons
     - Operation: ReLU activation followed by dropout with a rate of 0.5
     
   - **Layer 2**: 
     - Input: 4096 neurons
     - Output: 1024 neurons
     - Operation: ReLU activation followed by dropout with a rate of 0.5
     
   - **Layer 3**: 
     - Input: 1024 neurons
     - Output: 256 neurons
     - Operation: ReLU activation followed by dropout with a rate of 0.5
     
   - **Output Layer**: 
     - Input: 256 neurons
     - Output: 3 neurons (corresponding to the three classes: COVID-19, Pneumonia, Normal)
     - Operation: No activation function (since we use CrossEntropyLoss which applies Softmax internally)
     
   These fully connected layers aggregate the learned features and perform the final classification into one of the three categories.

### 3. **Regularization**:
   - **Dropout**: Dropout is applied after each convolutional layer and fully connected layer with a rate ranging from 0.2 to 0.5 to prevent overfitting during training.

### 4. **Activation Function**:
   - **ReLU (Rectified Linear Unit)** is used after each convolutional and fully connected layer, helping the model learn non-linear patterns and reducing the likelihood of vanishing gradients.

### 5. **Output**:
   - The model outputs a probability distribution over the three classes for each image, where the class with the highest probability is selected as the predicted label.

## Results

The model was trained for **7 epochs** using the **Adam optimizer** with a learning rate of 0.0001 and **CrossEntropyLoss**. The following results were obtained:

- **Train Accuracy**: 96.87%
- **Test Accuracy**: 95.34%



<img width="815" alt="image" src="https://github.com/user-attachments/assets/0bd38e41-e944-4258-a4b8-0b5f8a40e0fa" />
