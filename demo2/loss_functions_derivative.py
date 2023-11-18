import numpy as np

def mse_derivative(y_true, y_pred):
    """
    Calculate the derivative of the Mean Squared Error (MSE) loss function.
    Useful in gradient descent for minimizing the MSE in regression problems.

    :param y_true: numpy.ndarray, the true values.
    :param y_pred: numpy.ndarray, the predicted values.
    :return: numpy.ndarray, the derivative of the MSE.
    """
    return 2 * (y_pred - y_true)


def mae_derivative(y_true, y_pred):
    """
    Calculate the derivative of the Mean Absolute Error (MAE) loss function.
    Useful in gradient descent, especially in regression problems with outliers.

    :param y_true: numpy.ndarray, the true values.
    :param y_pred: numpy.ndarray, the predicted values.
    :return: numpy.ndarray, the derivative of the MAE.
    """
    return np.where(y_pred > y_true, 1, np.where(y_pred < y_true, -1, 0))


def binary_cross_entropy_derivative(y_true, y_pred):
    """
    Calculate the derivative of the Binary Cross-Entropy loss function.
    Useful in gradient descent for binary classification problems.

    :param y_true: numpy.ndarray, the true binary values.
    :param y_pred: numpy.ndarray, the predicted probability values.
    :return: numpy.ndarray, the derivative of the Binary Cross-Entropy.
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


def categorical_cross_entropy_derivative(y_true, y_pred):
    """
    Calculate the derivative of the Categorical Cross-Entropy loss function.
    Useful in gradient descent for multi-class classification problems.

    :param y_true: numpy.ndarray, the true label vectors (one-hot encoded).
    :param y_pred: numpy.ndarray, the predicted probability distributions.
    :return: numpy.ndarray, the derivative of the Categorical Cross-Entropy.
    """
    return y_pred - y_true
