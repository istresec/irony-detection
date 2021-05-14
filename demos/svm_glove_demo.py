import argparse
import importlib
from pathlib import Path

from podium.vectorizers import GloVe
from sklearn.metrics import accuracy_score

from benchmark_system.example import parse_dataset, featurize
from models.basic_model import BasicModel
from util.utils import load_and_preprocess


def get_embedding_format(x, vocab, glove):
    embeddings = glove.load_vocab(vocab)
    x = embeddings[x[:]]
    x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

    return x


# Test SVM with basic model.
if __name__ == '__main__':
    conf_path = Path("..\configs\svm.py")

    # Get configs
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Initialize GloVe
    glove = GloVe(dim=50)

    # Initialize model
    model = BasicModel(conf.backbone)

    print("\nUsing Podium:")

    # Without punctuation
    conf.remove_punctuation = True
    x, y, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
    x = get_embedding_format(x, vocab, glove)
    x_test = get_embedding_format(x_test, vocab, glove)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_test = model.predict(x_test)
    acc = accuracy_score(y_hat, y.ravel())
    acc_test = accuracy_score(y_hat_test, y_test.ravel())
    print(f"Accuracy on the train set (without punctuation): {acc:.4f}")
    print(f"Accuracy on the test set (without punctuation): {acc_test:.4f}")

    # With punctuation
    conf.remove_punctuation = False
    x, y, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
    x = get_embedding_format(x, vocab, glove)
    x_test = get_embedding_format(x_test, vocab, glove)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_test = model.predict(x_test)
    acc = accuracy_score(y_hat, y.ravel())
    acc_test = accuracy_score(y_hat_test, y_test.ravel())
    print(f"Accuracy on the train set (with punctuation): {acc:.4f}")
    print(f"Accuracy on the test set (with punctuation): {acc_test:.4f}")

    # Benchmark system
    print("\nUsing benchmark:")
    dataset_path = '../datasets/train/SemEval2018-T3-train-task' + conf.test_task
    dataset_path += '_emoji' if conf.test_emojis else ''
    dataset_path += '_ironyHashtags' if conf.test_irony_hashtags else ''
    dataset_path += '.txt'

    corpus, y = parse_dataset(dataset_path)
    x = featurize(corpus)

    model.fit(x, y)
    y_hat = model.predict(x)
    acc = accuracy_score(y_hat, y)
    print(f"Accuracy on the train set with benchmark system (tf-idf BOW): {acc:.4f}")
