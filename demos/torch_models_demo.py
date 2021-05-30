from util.utils import load_and_preprocess, update_stats, logger, calculate_statistics, train, evaluate
from models.simple_rnn import RNNClassifier
from util.dataloader import PytorchDataset, PytorchFeatureDataset
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import torch.nn as nn
import importlib
import torch
import numpy as np

if __name__ == '__main__':
    # Define configuration path
    conf_path = Path("../configs/cnn_lstm.py")

    # Get configuaration
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Set seeds for reproducibility
    seed = conf.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # Loading data
    feature_dim = 0
    if not conf.use_features:
        x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
        train_dataset = PytorchDataset(x, y)
        valid_dataset = PytorchDataset(x_val, y_val)
        test_dataset = PytorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)
    else:
        x, y, x_val, y_val, x_test, y_test, vocab, data, data_v, data_t = load_and_preprocess(conf, padding=True)
        features = np.array([feature for feature in data]).transpose()
        features_val = np.array([feature for feature in data_v]).transpose()
        features_test = np.array([feature for feature in data_t]).transpose()

        feature_dim = features.shape[1]

        train_dataset = PytorchFeatureDataset(x, features, y)
        valid_dataset = PytorchFeatureDataset(x_val, features_val, y_val)
        test_dataset = PytorchFeatureDataset(x_test, features_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

    max_length = x.shape[1]

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = glove.load_vocab(vocab)
    embed_dim = conf.glove_dim
    padding_index = vocab.get_padding_index()
    embedding_matrix = nn.Embedding(len(vocab), embed_dim, padding_idx=padding_index)

    # Copy the pretrained GloVe word embeddings
    embedding_matrix.weight.data.copy_(torch.from_numpy(embeddings))

    # Model without punctuation
    model = conf.model_constructor(embedding_matrix, max_length, feature_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # Train
    train(model, train_dataloader, valid_dataloader, optimizer, criterion, device, path, num_labels=2,
          epochs=conf.epochs, batch_size=conf.batch_size, early_stopping=conf.early_stopping,
          early_stop_tolerance=conf.early_stop_tolerance, features=conf.use_features)
    torch.save(model, path / "model.pth")

    # Testing the best model
    model = torch.load(path / "best_model.pth")
    model.to(device)
    loss, acc, conf_matrix = evaluate(model, test_dataloader, device, criterion, num_labels=2,
                                      features=conf.use_features)
    acc_percentage = acc / len(test_dataloader) / conf.batch_size
    precision, recall, f1 = calculate_statistics(conf_matrix)
    print("[Test Stats - Best model]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%"
          .format(loss / len(test_dataloader), acc / len(test_dataloader) / conf.batch_size * 100, f1 * 100))
    logger(log_path_test, acc_percentage, f1, precision, recall)

    # Testing the model
    model = torch.load(path / "model.pth")
    model.to(device)
    loss, acc, conf_matrix = evaluate(model, test_dataloader, device, criterion, num_labels=2,
                                      features=conf.use_features)
    acc_percentage = acc / len(test_dataloader) / conf.batch_size
    precision, recall, f1 = calculate_statistics(conf_matrix)
    print("[Test Stats - Final model]: loss = {:.3f}, acc = {:.3f}%, f1 = {:.3f}%"
          .format(loss / len(test_dataloader), acc / len(test_dataloader) / conf.batch_size * 100, f1 * 100))
    logger(log_path_test, acc_percentage, f1, precision, recall)
