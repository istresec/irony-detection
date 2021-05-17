from util.utils import load_and_preprocess, update_stats, logger, calculate_statistics
from models.simple_rnn import RNNClassifier
from util.dataloader import PytorchDataset
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import torch.nn as nn
import numpy as np
import importlib
import torch
import tqdm


def train(model, data, data_valid, optimizer, criterion, num_labels=2, best_accuracy=0):
    model.train()
    loss_t = 0
    accuracy, confusion_matrix = 0, np.zeros((num_labels, num_labels), dtype=int)
    for batch_idx, (x, y) in tqdm.tqdm(enumerate(data), total=len(data)):
        model.zero_grad()
        x, y = x.to(device), y.squeeze().to(device)
        lens = []
        for i in range(x.shape[0]):
            lens.append(x[i].shape[0])
        logits = model(x, sorted(lens))
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
        loss = criterion(logits, y)
        loss_t += loss.item()
        loss.backward()
        optimizer.step()
    _, _, f1 = calculate_statistics(confusion_matrix)
    print("[Train Stats]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%".format(loss_t/len(data),
                                                                             accuracy/len(data)/conf.batch_size * 100,
                                                                             f1*100))
    model.eval()
    with torch.no_grad():
        loss_v = 0
        accuracy_v, confusion_matrix_v = 0, np.zeros((num_labels, num_labels), dtype=int)
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(data_valid), total=len(data_valid)):
            x, y = x.to(device), y.squeeze().to(device)
            lens = []
            for i in range(x.shape[0]):
                lens.append(x[i].shape[0])
            logits = model(x, sorted(lens))
            accuracy_v, confusion_matrix_v = update_stats(accuracy_v, confusion_matrix_v, logits, y)
            loss = criterion(logits, y)
            loss_v += loss.item()
    _, _, f1 = calculate_statistics(confusion_matrix_v)
    print("[Valid Stats]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%".format(loss_v/len(data_valid),
                                                                             accuracy_v/len(data_valid)/conf.batch_size * 100,
                                                                             f1*100))

    if accuracy_v > best_accuracy:
        torch.save(model, path / "best_model.pth")
    return accuracy, confusion_matrix, accuracy_v, confusion_matrix_v


def test(model, data, num_labels=2):
    model.eval()
    accuracy, confusion_matrix = 0, np.zeros((num_labels, num_labels), dtype=int)
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(data), total=len(data)):
            x, y = x.to(device), y.squeeze().to(device)
            lens = []
            for i in range(len(x)):
                lens.append(len(x[i]))
            logits = model(x, sorted(lens))
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    _, _, f1 = calculate_statistics(confusion_matrix)
    print("[Test Stats]: acc = {:.3f}%, f1 = {:.3f}%".format(accuracy/len(data)/conf.batch_size * 100, f1*100))
    return accuracy, confusion_matrix


if __name__ == '__main__':
    conf_path = Path("..\configs\simple_rnn.py")

    # Get configs
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Save options
    path = Path(conf.save_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    log_path_train = path / "logs_train.txt"
    log_path_valid = path / "logs_valid.txt"
    log_path_test = path / "logs_test.txt"

    logger(log_path_train, "Accuracy", "F1", "Precision", "Recall")
    logger(log_path_valid, "Accuracy", "F1", "Precision", "Recall")
    logger(log_path_test, "Accuracy", "F1", "Precision", "Recall")

    # Initialize GloVe
    glove = GloVe(dim=conf.glove_dim)

    print("\nUsing Podium:")

    # Without punctuation
    conf.remove_punctuation = False

    # Loading data
    x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
    train_dataset = PytorchDataset(x, y)
    valid_dataset = PytorchDataset(x_val, y_val)
    test_dataset = PytorchDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = glove.load_vocab(vocab)
    embed_dim = conf.glove_dim
    padding_index = vocab.get_padding_index()
    embedding_matrix = nn.Embedding(len(vocab), embed_dim,
                                    padding_idx=padding_index)

    # Copy the pretrained GloVe word embeddings
    embedding_matrix.weight.data.copy_(torch.from_numpy(embeddings))

    model = RNNClassifier(embedding_matrix, embed_dim=conf.glove_dim, hidden_dim=conf.hidden_dim,
                          num_layers=conf.num_layers, dropout=conf.dropout, num_labels=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # Train loop
    acc = 0
    for epoch in range(conf.epochs):
        acc, conf_matrix, acc_v, conf_matrix_v = train(model, train_dataloader, valid_dataloader, optimizer,
                                                       criterion, num_labels=2, best_accuracy=acc)
        acc_percentage = acc / len(train_dataloader) / conf.batch_size
        precision, recall, f1 = calculate_statistics(conf_matrix)
        acc_percentage_v = acc_v / len(valid_dataloader) / conf.batch_size
        precision_v, recall_v, f1_v = calculate_statistics(conf_matrix_v)
        logger(log_path_train, acc_percentage, f1, precision, recall)
        logger(log_path_valid, acc_percentage_v, f1_v, precision_v, recall_v)

    torch.save(model, path / "model.pth")

    # Testing the model
    model = torch.load(path / "best_model.pth")
    model.to(device)
    acc, conf_matrix = test(model, test_dataloader, num_labels=2)
    acc_percentage = acc / len(test_dataloader) / conf.batch_size
    precision, recall, f1 = calculate_statistics(conf_matrix)
    logger(log_path_test, acc_percentage, f1, precision, recall)

    # no punctuation 99.109% on train and 64.668% on test
    # with punctuation 98.900% on train and 52.551% on test
