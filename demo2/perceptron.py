import numpy as np

from activation_functions import sigmoid
from activation_functions_derivative import sigmoid_derivative
from loss_functions import mse_loss
from loss_functions_derivative import mse_derivative


# Define the input and output for the AND truth table
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
outputs = np.array([[0], [1], [1], [1]])

# Initialize weights with a small random value
np.random.seed()
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
learning_rate = 0.4

# Define a function to predict the output using the trained weights and bias
def predict(input_data):
    weighted_sum = np.dot(input_data, weights) + bias
    activated_output = sigmoid(weighted_sum)
    return activated_output

# Display the weights and bias before training
print("Weights before training:")
print(weights)
print("\nBias before training:")
print(bias)

# Train the neural network
for epoch in range(40):
    # Get model predictions
    input_layer = inputs
    activated_output = predict(input_layer)
    
    # Get the loss estimate
    loss = mse_loss(activated_output, outputs)

    # Calculate the error
    error_derivative = mse_derivative(activated_output, outputs)
    
    # Adjust weights and bias using backpropagation
    adjustments = error_derivative * sigmoid_derivative(activated_output)
    weights += np.dot(input_layer.T, adjustments) * learning_rate
    bias += np.sum(adjustments, axis=0) * learning_rate

    # Display the weights and bias after training
    print(f"\nWeight [epoch = {epoch}]:")
    print(weights)
    print(f"Bias after [epoch = {epoch}]:")
    print(bias)
    print(f"Loss [epoch = {epoch}]:")
    print(loss)

    # Test the neural network with a new input
    print("\nTesting the trained neural network:")
    for i, input_data in enumerate(inputs):
        activated_output = predict(input_data)

        print(f"Input: {input_data} Output: {1 if activated_output > 0.5 else 0}")
