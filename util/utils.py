from util.dataloader import load_train_data, load_test_data
from pipeline.preprocessing import preprocess_and_tokenize
from podium.vocab import PAD
from pathlib import Path

import numpy as np
import importlib
import torch


def load_and_preprocess(config, padding=False):
    """
    Loads and preprocesses the train and test dataset based on given configuration.

    :param config: Configuration containing all necessary function arguments.
    :param padding: Determines if data is paddeed or not, False by default. Boolean.
    :return: The training data and labels x and y, the test data and labels x_test and y_test, and the vocab.
    """
    train_data, valid_data = load_train_data(config.test_task, emojis=config.test_emojis,
                                             irony_hashtags=config.test_irony_hashtags, split=True)
    test_data = load_test_data(config.test_task, emojis=config.test_emojis)

    train_dataset, vocab = preprocess_and_tokenize(train_data, remove_punct=config.remove_punctuation)
    test_dataset = preprocess_and_tokenize(test_data, remove_punct=config.remove_punctuation, use_vocab=False)
    valid_dataset = preprocess_and_tokenize(valid_data, remove_punct=config.remove_punctuation, use_vocab=False)

    x, y = train_dataset.batch(add_padding=padding)
    x_v, y_v = valid_dataset.batch()
    x_t_p, y_t = test_dataset.batch()

    if padding:
        required_length = x.shape[1]
        padding = [PAD()]
        for tweet in x_t_p:
            tweet_len = len(tweet)
            if tweet_len < required_length:
                tweet += padding * (required_length - tweet_len)
        for tweet in x_v:
            tweet_len = len(tweet)
            if tweet_len < required_length:
                tweet += padding * (required_length - tweet_len)
        # TODO: BUG -- if using padding and when *not* removing punctuation (remove_punctuation=False)
        # TODO: dataset.batch method returns a float type numpy array instead of an integer type array
        # TODO: remove if bug found
        x = x.astype(int)
    else:
        x = np.array(x, dtype=object)
        y = np.array(y)

    x_t = np.array([vocab.numericalize(tweet) for tweet in x_t_p], dtype=object if not padding else None)
    y_t = np.array(y_t)

    x_val = np.array([vocab.numericalize(tweet) for tweet in x_v], dtype=object if not padding else None)
    y_val = np.array(y_v)

    return x, y, x_val, y_val, x_t, y_t, vocab


def update_stats(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))
    for j, i in zip(max_ind, y):
        confusion_matrix[int(i), int(j)] += 1
    return accuracy + correct, confusion_matrix


def logger(path, accuracy, f1, precision, recall):
    with path.open(mode="a") as f:
        f.write(f"{accuracy} {f1} {precision} {recall}\n")


def calculate_statistics(conf_matrix):
    tp = np.diag(conf_matrix)[1]
    precision = tp / (tp + conf_matrix[0, 1])
    recall = tp / (tp + conf_matrix[1, 0])
    f1 = 2*precision*recall / (precision + recall)
    return precision, recall, f1


# test
if __name__ == '__main__':
    # Get configs
    conf_path = Path("../configs/no_model_default.py")
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    print('Without padding:')
    x, y, x_t, y_t, _ = load_and_preprocess(conf, padding=False)
    print(f"Shapes | x: {x.shape}, y: {y.shape}")
    print(f"Shapes | x_t: {x_t.shape}, y_t: {y_t.shape}")

    print('With padding:')
    x, y, x_t, y_t, _ = load_and_preprocess(conf, padding=True)
    print(f"Shapes | x: {x.shape}, y: {y.shape}")
    print(f"Shapes | x_t: {x_t.shape}, y_t: {y_t.shape}")
