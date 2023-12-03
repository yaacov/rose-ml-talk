import numpy as np

# Define the classification function
def classify(x):
    if x > 0.5:
        return 1
    else:
        return 0

# Vectorize the classify function
vectorized_classify = np.vectorize(classify)

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss and its derivative
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.random.randn(1)
        self.learning_rate = learning_rate

    # Predict function
    def predict(self, input_data):
        weighted_sum = np.dot(input_data, self.weights) + self.bias
        activated_output = sigmoid(weighted_sum)
        return activated_output

    # Train function
    def train(self, input_data, output_data, epochs):
        for epoch in range(epochs):
            # Forward pass
            activated_output = self.predict(input_data)

            # Compute loss
            loss = mse_loss(output_data, activated_output)

            # Backward pass
            error_derivative = mse_derivative(activated_output, output_data)
            adjustments = error_derivative * sigmoid_derivative(activated_output)

            # Update weights and bias
            self.weights += np.dot(input_data.T, adjustments) * self.learning_rate
            self.bias += np.sum(adjustments, axis=0) * self.learning_rate

            # Classification, using "cut of" at 0.5
            classes =  vectorized_classify(activated_output)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}")
                print(f"Loss: {loss}")
                print(f"Weights: {self.weights.T}")
                print(f"Bias: {self.bias}")
                print(f"{classes.T} \n---")


# Define the input and output for a truth table
learning_rate = 0.8
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
outputs = np.array([[0], [0], [0], [1]])

perceptron = Perceptron(2, learning_rate)
perceptron.train(inputs, outputs, epochs=200)
