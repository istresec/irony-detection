import sys

from util.dataloader import load_train_data, load_test_data
from pipeline.preprocessing import preprocess_and_tokenize
from podium.vocab import PAD
from pathlib import Path

import numpy as np
import importlib
import torch
import tqdm


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
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def train(model, data, data_valid, optimizer, criterion, device, path, num_labels=2, epochs=100,
          batch_size=32, early_stopping=True, early_stop_tolerance=5, max_norm=0.25):
    log_path_train = path / "logs_train.txt"
    log_path_valid = path / "logs_valid.txt"

    early_stop_ctr, best_f1 = 0, 0
    for epoch in range(1, epochs + 1):
        # Train step
        model.train()
        loss_t, accuracy_t, conf_mat_t = 0, 0, np.zeros((num_labels, num_labels), dtype=int)
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(data), total=len(data)):
            model.zero_grad()
            x, y = x.to(device), y.squeeze().to(device)
            lens = []
            for i in range(x.shape[0]):
                lens.append(x[i].shape[0])
            logits = model(x, sorted(lens))
            accuracy_t, conf_mat_t = update_stats(accuracy_t, conf_mat_t, logits, y)
            loss = criterion(logits, y)
            loss_t += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        prec_t, recall_t, f1_t = calculate_statistics(conf_mat_t)
        print("[Train Stats]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%"
              .format(loss_t / len(data), accuracy_t / len(data) / batch_size * 100, f1_t * 100))
        logger(log_path_train, accuracy_t, f1_t, prec_t, recall_t)

        # Evaluate validation
        loss_v, acc_v, conf_mat_v = evaluate(model, data_valid, device, criterion, num_labels=num_labels)
        prec_v, recall_v, f1_v = calculate_statistics(conf_mat_v)
        print("[Valid Stats]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%"
              .format(loss_v / len(data_valid), acc_v / len(data_valid) / batch_size * 100, f1_v * 100))
        logger(log_path_valid, acc_v, f1_v, prec_v, recall_v)

        # Early stopping and best model saving
        if f1_v > best_f1:
            best_f1 = f1_v
            torch.save(model, path / "best_model.pth")
            early_stop_ctr = 0
        else:
            early_stop_ctr += 1
            if early_stopping and (early_stop_ctr > early_stop_tolerance):
                print(f"Early stopping at epoch {epoch} -- model did not improve in {early_stop_tolerance} steps.")
                break


def evaluate(model, data, device, criterion, num_labels=2):
    model.eval()
    loss, accuracy, confusion_matrix = 0, 0, np.zeros((num_labels, num_labels), dtype=int)
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(data), total=len(data)):
            x, y = x.to(device), y.squeeze().to(device)
            lens = []
            for i in range(len(x)):
                lens.append(len(x[i]))
            logits = model(x, sorted(lens))
            loss += criterion(logits, y).item()
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    return loss, accuracy, confusion_matrix


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
