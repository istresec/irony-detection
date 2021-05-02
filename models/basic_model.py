import argparse
import importlib
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np

from models.model import Model
from pipeline.preprocessing import preprocess_and_tokenize, tf_idf_vectorization
from util import dataloader


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


"""
# Indices to one-hot encoding over the entire dataset - does not work.
def to_one_hot(x, vector_len):
    output = np.zeros((x.shape[0], x.shape[1], vector_len), dtype=int)

    for row in range(x.shape[0]):
        output[row] = np.eye(vector_len)[x[row]]

    return output
"""

# Test SVM with basic model.
if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Config file from which model and dataset is read")
    args = parser.parse_args()
    conf_path = Path(args.config)

    # Get configs
    spec = importlib.util.spec_from_file_location("module", conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Get data
    train_data = dataloader.load_train_data(conf.test_task, emojis=conf.test_emojis,
                                            irony_hashtags=conf.test_irony_hashtags)
    train_dataset = preprocess_and_tokenize(train_data, remove_punct=conf.remove_punctuation)
    x, y = train_dataset.batch(add_padding=True)
    tfidf_batch = tf_idf_vectorization(train_dataset, x)

    # Train model via tf-idf
    model = BasicModel(conf.backbone)
    model.fit(tfidf_batch, y.ravel())
    y_hat = model.predict(tfidf_batch)
    acc = accuracy_score(y_hat, y.ravel())
    print(f"Accuracy on the train set: {acc:.4f}")

    # Train model via dictionary indices (instead of one-hot encoding :( )
    model = BasicModel(conf.backbone)
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    acc = accuracy_score(y_hat, y.ravel())
    print(f"Accuracy on the train set: {acc:.4f}")
