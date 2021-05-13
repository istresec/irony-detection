import argparse
import importlib
from pathlib import Path

from sklearn.metrics import accuracy_score

from benchmark_system.example import parse_dataset, featurize
from models.basic_model import BasicModel
from pipeline.preprocessing import preprocess_and_tokenize, tf_idf_vectorization
from podium.vectorizers import GloVe
from util import dataloader

import numpy as np


def load_and_preprocess(conf, glove, remove_punct):

    # Load and preprocess data - with punctuation
    train_data = dataloader.load_train_data(conf.test_task, emojis=conf.test_emojis,
                                            irony_hashtags=conf.test_irony_hashtags)
    test_data = dataloader.load_test_data(conf.test_task, emojis=conf.test_emojis)
    train_dataset, vocab = preprocess_and_tokenize(train_data, finalize=False, remove_punct=remove_punct)
    test_dataset, _ = preprocess_and_tokenize(test_data, finalize=False, remove_punct=remove_punct)
    train_dataset.finalize_fields(train_dataset, test_dataset)  # use both datasets to get complete Vocab for GloVe
    test_dataset.finalize_fields()
    x, y = train_dataset.batch(add_padding=True)
    x = x.astype(int)
    x_test, y_test = test_dataset.batch(add_padding=True)

    # Handle different padding
    x_test_p = np.ones((x_test.shape[0], x.shape[1]), dtype=np.integer)
    x_test_p[:, :x_test.shape[1]] = x_test
    x_test = x_test_p

    # Get embedding formats
    embeddings = glove.load_vocab(vocab)
    x = embeddings[x[:]]
    x_test = embeddings[x_test[:]]
    x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    return x, y, x_test, y_test


# Test SVM with basic model.
if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help="Path to config file from which model and dataset is read")
    args = parser.parse_args()
    conf_path = Path(args.config)
    # This works too
    # conf_path = Path("..\configs\svm.py")

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
    x, y, x_test, y_test = load_and_preprocess(conf, glove, True)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_test = model.predict(x_test)
    acc = accuracy_score(y_hat, y.ravel())
    acc_test = accuracy_score(y_hat_test, y_test.ravel())
    print(f"Accuracy on the train set (without punctuation): {acc:.4f}")
    print(f"Accuracy on the test set (without punctuation): {acc_test:.4f}")

    # With punctuation
    x, y, x_test, y_test = load_and_preprocess(conf, glove, False)

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
