# House Price Prediction using Neural Networks

This project demonstrates the implementation of neural network models to predict house prices using TensorFlow/Keras. The project explores different neural network architectures and techniques to improve model performance.

## Project Overview

The project uses a dataset containing house price information with 10 input features. Three different neural network models are implemented to compare their performance:

1. Simple Neural Network
   - 2 hidden layers with 32 neurons each
   - ReLU activation
   - SGD optimizer

2. Deep Neural Network
   - 4 hidden layers with 1000 neurons each
   - ReLU activation
   - Adam optimizer

3. Deep Neural Network with Dropout
   - 4 hidden layers with 1000 neurons each
   - Dropout layers (0.5) after each hidden layer
   - ReLU activation
   - Adam optimizer

## Dependencies

- pandas
- scikit-learn
- tensorflow/keras
- matplotlib

## Dataset

The project uses `housepricedata.csv` which contains house-related features. The data is preprocessed using MinMaxScaler for normalization.

## Model Architecture

All models are trained with:

- Binary cross-entropy loss
- Accuracy metric
- 100 epochs
- Batch size of 32
- Train/Validation/Test split (70%/15%/15%)

## Visualizations

The project includes visualization of:

- Model loss curves (training and validation)
- Model accuracy curves (training and validation)

These visualizations help in understanding model performance and identifying potential overfitting/underfitting issues.

## Usage

Run the Jupyter notebook `housepricedata_NN.ipynb` to:

1. Load and preprocess the data
2. Train the three different models
3. Visualize the results and compare model performance
