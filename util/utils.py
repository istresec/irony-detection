from util.dataloader import load_train_data, load_test_data
from pipeline.preprocessing import preprocess_and_tokenize
from podium.vocab import PAD
from pathlib import Path

import numpy as np
import pandas as pd
import importlib
import torch
import tqdm


def load_and_preprocess(config, padding=False):
    """
    Loads and preprocesses the train and test dataset based on given configuration.

    :param config: Configuration containing all necessary function arguments.
    :param padding: Determines if data is padded or not, False by default. Boolean.
    :return: The training data and labels x and y, the test data and labels x_test and y_test, and the vocab.
    """
    train_data, valid_data = load_train_data(config.test_task, emojis=config.test_emojis,
                                             irony_hashtags=config.test_irony_hashtags, split=True)
    test_data = load_test_data(config.test_task, emojis=config.test_emojis)

    train_dataset, vocab = preprocess_and_tokenize(train_data, remove_punct=config.remove_punctuation,
                                                   use_features=config.use_features)
    valid_dataset = preprocess_and_tokenize(valid_data, remove_punct=config.remove_punctuation, use_vocab=False,
                                            use_features=config.use_features)
    test_dataset = preprocess_and_tokenize(test_data, remove_punct=config.remove_punctuation, use_vocab=False,
                                           use_features=config.use_features)

    # Data now could contain additional punctuation fields
    data = train_dataset.batch()
    x, y = data.pop('input_text'), data.pop('target')
    data_v = valid_dataset.batch()
    # TODO: y_v is flipped, fixed below, batch is bugged as hell yo ;;)))))))))
    x_v, y_v = data_v.pop('input_text'), data_v.pop('target')
    data_t = test_dataset.batch()
    x_t, y_t = data_t.pop('input_text'), data_t.pop('target')

    if padding:
        required_length = max([len(i) for i in x] + [len(i) for i in x_v] + [len(i) for i in x_t])
        padding = [PAD()]
        for i in range(len(x)):
            tweet_len = len(x[i])
            if required_length > tweet_len:
                x[i] = np.concatenate((x[i], np.array(vocab.numericalize(padding * (required_length - tweet_len)))))
        for tweet in x_v:
            tweet_len = len(tweet)
            if required_length > tweet_len:
                tweet += padding * (required_length - tweet_len)
        for tweet in x_t:
            tweet_len = len(tweet)
            if required_length > tweet_len:
                tweet += padding * (required_length - tweet_len)

    x = np.array(x, dtype=object if not padding else int)
    if y[0] != train_dataset[0].target[1]:
        y = (np.array(y) + 1) % 2
    else:
        y = np.array(y)

    x_v = np.array([vocab.numericalize(tweet) for tweet in x_v], dtype=object if not padding else int)
    # TODO: fix for error above
    if y_v[0] != valid_dataset[0].target[1]:
        y_v = (np.array(y_v) + 1) % 2
    else:
        y_v = np.array(y_v)

    x_t = np.array([vocab.numericalize(tweet) for tweet in x_t], dtype=object if not padding else int)
    if y_t[0] != test_dataset[0].target[1]:
        y_t = (np.array(y_t) + 1) % 2
    else:
        y_t = np.array(y_t)

    ret_data = (x, y, x_v, y_v, x_t, y_t, vocab) if not config.use_features \
        else (x, y, x_v, y_v, x_t, y_t, vocab, data, data_v, data_t)

    return ret_data


def load_preprocess_one_example(config, example, device):
    """
    Loads and preprocesses the train and test dataset based on given configuration.

    :param config: Configuration containing all necessary function arguments.
    :return: The training data and labels x and y, the test data and labels x_test and y_test, and the vocab.
    """
    train_data, valid_data = load_train_data(config.test_task, emojis=config.test_emojis,
                                             irony_hashtags=config.test_irony_hashtags, split=True)

    train_dataset, vocab = preprocess_and_tokenize(train_data, remove_punct=True,
                                                   use_features=False)
    train_dataset, vocab2 = preprocess_and_tokenize(train_data, remove_punct=False,
                                                   use_features=False)
    df_example = pd.DataFrame(data={"text": [example], "label": [0]})

    dataset_no_punct = preprocess_and_tokenize(df_example, use_vocab=False, remove_punct=True, use_features=False)
    dataset_punct = preprocess_and_tokenize(df_example, use_vocab=False, remove_punct=False, use_features=False)
    dataset_punct_f = preprocess_and_tokenize(df_example, use_vocab=False, remove_punct=False, use_features=True)

    data_np = dataset_no_punct.batch()
    x_np, y_np = data_np.pop('input_text'), data_np.pop('target')

    data_p = dataset_punct.batch()
    x_p, y_p = data_p.pop('input_text'), data_p.pop('target')

    data_pf = dataset_punct_f.batch()
    x_pf, y_pf = data_pf.pop('input_text'), data_pf.pop('target')

    x_np = torch.tensor(np.array([vocab.numericalize(tweet) for tweet in x_np], dtype=int), dtype=torch.long).to(device)
    x_p = torch.tensor(np.array([vocab2.numericalize(tweet) for tweet in x_p], dtype=int), dtype=torch.long).to(device)
    x_pf = torch.tensor(np.array([vocab2.numericalize(tweet) for tweet in x_pf], dtype=int), dtype=torch.long).to(device)

    return x_np, x_p, x_pf, data_pf


def update_stats(accuracy, confusion_matrix, logits, y):
    max_ind = get_predictions(logits)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))
    for i, j in zip(y, max_ind):
        confusion_matrix[int(i), int(j)] += 1
    return accuracy + correct, confusion_matrix


def get_predictions(logits):
    _, max_ind = torch.max(logits, 1)
    return max_ind


def logger(path, accuracy, f1, precision, recall):
    with path.open(mode="a") as f:
        f.write(f"{accuracy} {f1} {precision} {recall}\n")


def calculate_statistics(conf_matrix):
    tp = np.diag(conf_matrix)[1]
    precision = tp / (tp + conf_matrix[0, 1] + 1e-15)
    recall = tp / (tp + conf_matrix[1, 0] + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    return precision, recall, f1


def train(model, data, data_valid, optimizer, criterion, device, path, num_labels=2, epochs=100,
          batch_size=32, early_stopping=True, early_stop_tolerance=5, max_norm=0.25, features=False):
    log_path_train = path / "logs_train.txt"
    log_path_valid = path / "logs_valid.txt"

    early_stop_ctr, best_f1 = 0, 0
    for epoch in range(1, epochs + 1):
        # Train step
        model.train()
        loss_t, accuracy_t, conf_mat_t = 0, 0, np.zeros((num_labels, num_labels), dtype=int)
        for batch_idx, batch in tqdm.tqdm(enumerate(data), total=len(data)):
            if not features:
                x, y = batch
                x, y = x.to(device), y.squeeze().to(device)
            else:
                x, f, y = batch
                x, f, y = x.to(device), f.to(device), y.squeeze().to(device)

            model.zero_grad()
            lens = get_lengths(x)
            # x, y, lens = clean_zero_values(x, y, lens, batch_size)

            if not features:
                logits = model(x, lens)
            else:
                logits = model(x, lens, f)

            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            loss_t += loss.item()
            accuracy_t, conf_mat_t = update_stats(accuracy_t, conf_mat_t, logits, y)

        prec_t, recall_t, f1_t = calculate_statistics(conf_mat_t)
        print("[Train Stats]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%"
              .format(loss_t / len(data), accuracy_t / len(data) / batch_size * 100, f1_t * 100))
        logger(log_path_train, accuracy_t, f1_t, prec_t, recall_t)

        # Evaluate validation
        loss_v, acc_v, conf_mat_v = evaluate(model, data_valid, device, criterion, num_labels=num_labels,
                                             features=features)
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


def evaluate(model, data, device, criterion, num_labels=2, features=False, return_predictions=False):
    model.eval()
    loss, accuracy, confusion_matrix = 0, 0, np.zeros((num_labels, num_labels), dtype=int)
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(data), total=len(data)):
            if not features:
                x, y = batch
                x, y = x.to(device), y.squeeze().to(device)
            else:
                x, f, y = batch
                x, f, y = x.to(device), f.to(device), y.squeeze().to(device)

            lens = get_lengths(x)
            # x, y, lens = clean_zero_values(x, y, lens, batch_size)

            if not features:
                logits = model(x, lens)
            else:
                logits = model(x, lens, f)
            loss += criterion(logits, y).item()
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
            if return_predictions:
                batch_predictions = get_predictions(logits)
                predictions.extend(batch_predictions.detach().cpu().numpy())
    if return_predictions:
        return loss, accuracy, confusion_matrix, predictions
    else:
        return loss, accuracy, confusion_matrix


def get_lengths(x):
    lens = []
    for i in range(x.shape[0]):
        idx_of_padding = (x[i] == 1).nonzero(as_tuple=True)[0]
        if len(idx_of_padding) > 0:
            idx_of_padding = idx_of_padding[0].detach().cpu().data.numpy().item()
        else:
            idx_of_padding = x.shape[1]
        lens.append(idx_of_padding)
    return lens


def clean_zero_values(x, y, lengths, batch_size):
    if 0 in lengths:
        idx_0 = lengths.index(0)
        if idx_0 == 0:
            x = x[1:]
            y = y[1:]
        elif idx_0 == batch_size - 1:
            x = x[:-1]
            y = y[:-1]
        else:
            x = torch.cat([x[:idx_0], x[idx_0 + 1:]])
            y = torch.cat([y[:idx_0], y[idx_0 + 1:]])
        lengths.remove(0)
    return x, y, lengths


# test
if __name__ == '__main__':
    # Get configs
    """
    conf_path = Path("../configs/no_model_default.py")
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    print('Without padding:')
    x, y, x_v, y_v, x_t, y_t, _ = load_and_preprocess(conf, padding=False)
    print(f"Shapes | x: {x.shape}, y: {y.shape}")
    print(f"Shapes | x_v: {x_v.shape}, y_v: {y_v.shape}")
    print(f"Shapes | x_t: {x_t.shape}, y_t: {y_t.shape}")

    print('With padding:')
    x, y, x_v, y_v, x_t, y_t, _ = load_and_preprocess(conf, padding=True)
    print(f"Shapes | x: {x.shape}, y: {y.shape}")
    print(f"Shapes | x_v: {x_v.shape}, y_v: {y_v.shape}")
    print(f"Shapes | x_t: {x_t.shape}, y_t: {y_t.shape}")
    """
    y = [0, 1, 0, 1]
    batch = torch.tensor([1, 0, 1, 0, 1])
    y.extend(batch.numpy())
    print(y)
