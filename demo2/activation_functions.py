import numpy as np

def relu(x):
    """
    ReLU (Rectified Linear Unit) Activation Function.
    
    It returns zero for any negative input, but for any positive input, it returns that value back.
    ReLU is often used in hidden layers as it helps with the vanishing gradient problem and allows for faster training.
    
    Parameters:
    x (float): The input value or an array of input values.
    
    Returns:
    float: The output after applying the ReLU function.
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid Activation Function.
    
    It maps any input value into a value between 0 and 1, which can be interpreted as a probability.
    It is commonly used for the output layer of a binary classification problem, where the result is interpreted as the probability of the input being in one class or the other.
    
    Parameters:
    x (float): The input value or an array of input values.
    
    Returns:
    float: The output after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax Activation Function.
    
    It is a generalization of the sigmoid function that squashes the output of each unit to be between 0 and 1, like a sigmoid, but also divides each output such that the total sum of the outputs is equal to 1.
    Softmax is often used in the output layer of a neural network that deals with multi-class classification problems, where it can show the probability distribution across different classes.
    
    Parameters:
    x (float): The input value or an array of input values.
    
    Returns:
    float: The output after applying the softmax function.
    """
    e_x = np.exp(x - np.max(x))  # subtract max to stabilize the computations
    return e_x / e_x.sum(axis=0)  # add the axis parameter to ensure it sums across the correct axis
