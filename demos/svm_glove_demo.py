import argparse
import importlib
from pathlib import Path

from podium.vectorizers import GloVe
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef

from benchmark_system.example import parse_dataset, featurize
from models.basic_model import BasicModel
from util.utils import load_and_preprocess

import numpy as np


def get_embedding_format(x, embeddings):
    x = [embeddings[entry] if len(entry) > 0 else embeddings[[0]] for entry in x]
    x = np.array([np.mean(entry, axis=0) for entry in x])
    return x


# Test SVM with basic model.
if __name__ == '__main__':
    # Set seeds for reproducibility
    seed = 8008135
    np.random.seed(seed)

    conf_path = Path("..\configs\svm.py")
    save_path = '../saves/svm/'

    # Get configs
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Initialize GloVe
    glove = GloVe(dim=300)

    # Initialize model
    model = BasicModel(conf.backbone)

    print("\nUsing Podium:")

    # Without punctuation
    conf.remove_punctuation = True
    conf.use_features = False
    x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=False)
    embeddings = glove.load_vocab(vocab)
    x = get_embedding_format(x, embeddings)
    x_val = get_embedding_format(x_val, embeddings)
    x_test = get_embedding_format(x_test, embeddings)
    y = np.array(y)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)
    f1 = f1_score(y, y_hat)
    f1_val = f1_score(y_val, y_hat_val)
    prec_test = precision_score(y_test, y_hat_test)
    recall_test = recall_score(y_test, y_hat_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)
    matthew_test = matthews_corrcoef(y_test, y_hat_test)
    f1_test = f1_score(y_test, y_hat_test)
    print(f"F1 on the train set (without punctuation): {f1:.4f}")
    print(f"F1 on the validation set (without punctuation): {f1_val:.4f}")
    print("--------------------------------------------------")
    print(f"F1 on the test set (without punctuation): {f1_test:.4f}")
    print(f"Acc. on the test set (without punctuation): {accuracy_test:.4f}")
    print(f"Prec. on the test set (without punctuation): {prec_test:.4f}")
    print(f"Rec. on the test set (without punctuation): {recall_test:.4f}")
    print(f"Matthew corr. on the test set (without punctuation): {matthew_test:.4f}\n")
    with open(save_path + 'no_punct.txt', 'w') as f:
        for test in y_hat_test:
            f.write(str(test) + '\n')

    # With punctuation
    conf.remove_punctuation = False
    conf.use_features = False
    x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=False)
    embeddings = glove.load_vocab(vocab)
    x = get_embedding_format(x, embeddings)
    x_val = get_embedding_format(x_val, embeddings)
    x_test = get_embedding_format(x_test, embeddings)
    y = np.array(y)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)
    f1 = f1_score(y, y_hat)
    f1_val = f1_score(y_val, y_hat_val)
    prec_test = precision_score(y_test, y_hat_test)
    recall_test = recall_score(y_test, y_hat_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)
    matthew_test = matthews_corrcoef(y_test, y_hat_test)
    f1_test = f1_score(y_test, y_hat_test)
    print(f"F1 on the train set (with punctuation): {f1:.4f}")
    print(f"F1 on the validation set (with punctuation): {f1_val:.4f}")
    print("--------------------------------------------------")
    print(f"F1 on the test set (with punctuation): {f1_test:.4f}")
    print(f"Acc. on the test set (with punctuation): {accuracy_test:.4f}")
    print(f"Prec. on the test set (with punctuation): {prec_test:.4f}")
    print(f"Rec. on the test set (with punctuation): {recall_test:.4f}")
    print(f"Matthew corr. on the test set (with punctuation): {matthew_test:.4f}\n")
    with open(save_path + 'punct.txt', 'w') as f:
        for test in y_hat_test:
            f.write(str(test) + '\n')

    # With punctuation and features
    conf.remove_punctuation = False
    conf.use_features = True
    x, y, x_val, y_val, x_test, y_test, vocab, data, data_v, data_t = load_and_preprocess(conf, padding=False)
    embeddings = glove.load_vocab(vocab)
    x = get_embedding_format(x, embeddings)
    features = np.array([feature for feature in data]).transpose()
    x = np.concatenate((x, features), axis=1)
    x_val = get_embedding_format(x_val, embeddings)
    features_val = np.array([feature for feature in data_v]).transpose()
    x_val = np.concatenate((x_val, features_val), axis=1)
    x_test = get_embedding_format(x_test, embeddings)
    features_test = np.array([feature for feature in data_t]).transpose()
    x_test = np.concatenate((x_test, features_test), axis=1)
    y = np.array(y)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Train model via embeddings
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)
    f1 = f1_score(y, y_hat)
    f1_val = f1_score(y_val, y_hat_val)
    prec_test = precision_score(y_test, y_hat_test)
    recall_test = recall_score(y_test, y_hat_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)
    matthew_test = matthews_corrcoef(y_test, y_hat_test)
    f1_test = f1_score(y_test, y_hat_test)
    print(f"F1 on the train set (with punctuation and features): {f1:.4f}")
    print(f"F1 on the validation set (with punctuation and features): {f1_val:.4f}")
    print("--------------------------------------------------")
    print(f"F1 on the test set (with punctuation and features): {f1_test:.4f}")
    print(f"Acc. on the test set (with punctuation and features): {accuracy_test:.4f}")
    print(f"Prec. on the test set (with punctuation and features): {prec_test:.4f}")
    print(f"Rec. on the test set (with punctuation and features): {recall_test:.4f}")
    print(f"Matthew corr. on the test set (with punctuation and features): {matthew_test:.4f}\n")
    with open(save_path + 'punct_w_features.txt', 'w') as f:
        for test in y_hat_test:
            f.write(str(test) + '\n')

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
