import numpy as np

def mse_loss(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) loss.
    Best used for regression problems with normally distributed errors.

    :param y_true: numpy.ndarray, the true values.
    :param y_pred: numpy.ndarray, the predicted values.
    :return: float, the MSE loss.
    """
    return np.mean(np.square(y_pred - y_true))


def mae_loss(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) loss.
    Best used for regression problems, particularly when handling outliers, 
    as it is less sensitive to them compared to MSE.

    :param y_true: numpy.ndarray, the true values.
    :param y_pred: numpy.ndarray, the predicted values.
    :return: float, the MAE loss.
    """
    return np.mean(np.abs(y_pred - y_true))


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Calculate the Binary Cross-Entropy loss.
    Best used for binary classification problems.

    :param y_true: numpy.ndarray, the true binary values.
    :param y_pred: numpy.ndarray, the predicted probability values.
    :return: float, the Binary Cross-Entropy loss.
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy_loss(y_true, y_pred):
    """
    Calculate the Categorical Cross-Entropy loss.
    Best used for multi-class classification problems.

    :param y_true: numpy.ndarray, the true label vectors (one-hot encoded).
    :param y_pred: numpy.ndarray, the predicted probability distributions.
    :return: float, the Categorical Cross-Entropy loss.
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))
