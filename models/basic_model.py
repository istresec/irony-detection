from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from models.model import Model


class BasicModel(Model):
    """
    Skeletal implementation of the Model abstract class.
    """

    def __init__(self, backbone):
        super().__init__(backbone)

    def fit(self, X, y=None, **params):
        return self.backbone.fit(X, y, **params)

    def predict(self, X, **params):
        return self.backbone.predict(X, **params)


# Simple test usage
if __name__ == '__main__':
    X, y = make_classification(n_features=4, random_state=11)
    model = BasicModel(LinearSVC())
    model.fit(X, y)
    y_hat = model.predict(X)
    acc = accuracy_score(y_hat, y)
    print(f"Accuracy on the train set: {acc:.4f}")