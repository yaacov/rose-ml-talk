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

# Neural Network Structure
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def predict(self, input_data):
        layer_output = input_data
        for w, b in zip(self.weights, self.biases):
            layer_output = sigmoid(np.dot(layer_output, w) + b)
        return layer_output

    def train(self, input_data, output_data, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward propagation
            layer_outputs = [input_data]
            for w, b in zip(self.weights, self.biases):
                layer_outputs.append(sigmoid(np.dot(layer_outputs[-1], w) + b))

            # Backpropagation
            loss = mse_loss(output_data, layer_outputs[-1])
            error = mse_derivative(output_data, layer_outputs[-1])

            for i in reversed(range(len(self.weights))):
                adjustments = error * sigmoid_derivative(layer_outputs[i+1])
                error = np.dot(adjustments, self.weights[i].T)
                self.weights[i] -= np.dot(layer_outputs[i].T, adjustments) * learning_rate
                self.biases[i] -= np.sum(adjustments, axis=0) * learning_rate

            # Classification, using "cut of" at 0.5
            classes = vectorized_classify(layer_outputs[-1])

            if epoch % 100 == 0:
                print(f"Epoch {epoch}")
                print(f"Loss: {loss}")
                print(f"Weights: {self.weights}")
                print(f"Bias: {self.biases}")
                print(f"{classes.T} \n---")

# Example usage
learning_rate = 0.8
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1])
nn.train(inputs, outputs, learning_rate, epochs=4000)
