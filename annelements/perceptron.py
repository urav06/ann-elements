import numpy as np

class Perceptron:
    """
    A class representing a single-layer perceptron.
    It can be used to perform binary classifications on linearly seperable data.
    """
    def __init__(self, nu: float = 0.01, weights: list = None, bias: int = None) -> None:
        self.nu = nu # Learning Rate
        self.weights = np.array(weights)
        self.bias = bias

        self.activation = lambda x: np.where(x>0, 1, 0) if isinstance(x, np.ndarray) else (1 if x>0 else 0)

    def learn(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        A learning algorithm to train the perceptron on given data.
        This function only returns if the training data is linearly seperabe.
        """
        _, dims = X.shape

        if self.weights is None:
            self.weights = np.ones(dims)
        if self.bias is None:
            self.bias = 0
        
        y = np.where(y>0, 1, 0) # Data Cleansing

        y_pred = self.classify(X)
        misclassifications = y_pred != y
        while np.any(misclassifications):
            for i, x_i in enumerate(X):
                if misclassifications[i]:
                    delta = self.nu * (y[i] - y_pred[i])
                    self.weights += delta * x_i # w_delta array of size (dims)
                    self.bias += delta

            y_pred = self.classify(X)
            misclassifications = y_pred != y

    def classify(self, X: np.ndarray) -> np.ndarray:
        fx = np.dot(X, self.weights) + self.bias
        gfx = self.activation(fx)
        return gfx
