def relu_derivative(x):
    """
    Derivative of the ReLU Activation Function.

    It calculates the derivative of the ReLU function which is used during the backpropagation process in neural networks.

    Parameters:
    x (float): The input value or an array of input values.

    Returns:
    float: The output after applying the derivative of the ReLU function.
    """
    return np.where(x > 0, 1.0, 0.0)

def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid Activation Function.

    It calculates the derivative of the sigmoid function which is used during the backpropagation process in neural networks.

    Parameters:
    x (float): The sigmoid value or an array of sigmoid values.

    Returns:
    float: The output after applying the derivative of the sigmoid function.
    """
    return x * (1 - x)

def softmax_derivative(x):
    """
    Derivative of the Softmax Activation Function.

    It calculates the derivative of the softmax function which is used during the backpropagation process in neural networks.

    Parameters:
    x (float): The softmax value or an array of softmax values.

    Returns:
    float: The output after applying the derivative of the softmax function.
    """
    return x * (1 - x)