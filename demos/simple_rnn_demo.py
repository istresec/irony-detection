from models.simple_rnn import RNNClassifier
from util.utils import load_and_preprocess
from util.dataloader import PytorchDataset
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import torch.nn as nn
import numpy as np
import importlib
import torch
import tqdm


def update_stats(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))
    for j, i in zip(max_ind, y):
        confusion_matrix[int(i),int(j)]+=1
    return accuracy + correct, confusion_matrix


def train(model, data, optimizer, criterion, num_labels=2):
    model.train()
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
        loss.backward()
        optimizer.step()
    print("[Train Accuracy]: {}/{} : {:.3f}%".format(
          accuracy, len(data) * conf.batch_size, accuracy / len(data) / conf.batch_size * 100))
    return accuracy, confusion_matrix


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
    print("[Test Accuracy]: {}/{} : {:.3f}%".format(
          accuracy, len(data) * conf.batch_size, accuracy / len(data) / conf.batch_size * 100))
    return accuracy, confusion_matrix


if __name__ == '__main__':
    conf_path = Path("..\configs\simple_rnn.py")

    # Get configs
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Initialize GloVe
    glove = GloVe(dim=conf.glove_dim)

    print("\nUsing Podium:")

    # Without punctuation
    conf.remove_punctuation = False

    # Loading data
    x, y, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
    train_dataset = PytorchDataset(x, y)
    test_dataset = PytorchDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
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

    model = RNNClassifier(embedding_matrix)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # Train loop
    for epoch in range(conf.epochs):
        train(model, train_dataloader, optimizer, criterion, num_labels=2)

    # Testing the model
    test(model, test_dataloader, num_labels=2)

    # no punctuation 99.109% on train and 64.668% on test
    # with punctuation 98.900% on train and 52.551% on test
