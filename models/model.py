from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract model class.
    """

    def __init__(self, backbone):
        """
        Initializes object with the model which will be used.

        :param backbone: Machine learning model
        """
        self.backbone = backbone

    @abstractmethod
    def fit(self, X, y=None, **params):
        """
        Fit model with given data.

        :param X: Data with which the model is trained
        :param y: Labels for the data in X
        :param params: Any extra params passed to backbone model predict function

        :return: Whatever the implementation of the backbone model returns
        """
        pass

    @abstractmethod
    def predict(self, X, **params):
        """
        Predict data labels.

        :param X: Data for which labels are being predicted
        :param params: Any extra params passed to backbone model predict function

        :return: Predicted labels
        """
        pass
