from tests import mcnemar

from util.dataloader import PytorchDataset, PytorchFeatureDataset
from util.utils import load_and_preprocess
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path

import torch.nn as nn
import numpy as np
import importlib
import torch

import matplotlib.pyplot as plt
import pandas as pd


def scatter_punct():
    interpunctions = features[:, -1]
    idx_sarcastic = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]
    plt.subplot(1, 2, 1)
    plt.hist(interpunctions[idx_non], color="pink")
    plt.title("Non sarcastic tweets")
    plt.subplot(1, 2, 2)
    plt.hist(interpunctions[idx_sarcastic], color="pink")
    plt.title("Sarcastic tweets")
    #plt.scatter(y_test, interpunctions)
    plt.show()


def check_svm():
    no_punct = pd.read_csv("C:/Users/Jelena/Downloads/no_punct(1).txt", header=None)
    punct = pd.read_csv("C:/Users/Jelena/Downloads/punct(1).txt", header=None)
    punct_features = pd.read_csv("C:/Users/Jelena/Downloads/punct_w_features(1).txt", header=None)
    table_1 = mcnemar.create_contingency_table(punct[0], no_punct[0])
    table_2 = mcnemar.create_contingency_table(punct[0], punct_features[0])
    mcnemar.make_tests(table_1, exact=True if (table_1[0, 1] + table_1[1, 0] < 25) else False,
                       correction=True, alpha=0.05)
    mcnemar.make_tests(table_2, exact=True if (table_2[0, 1] + table_2[1, 0] < 25) else False,
                       correction=True, alpha=0.05)


if __name__ == '__main__':
    # Define configuration path
    conf_path = Path("../saves/TAR_models/LSTM_DNN/punct/config.py")

    # Get configuaration
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    print(conf.test_mode)

    # Set seeds for reproducibility
    seed = conf.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize GloVe
    glove = GloVe(dim=conf.glove_dim)

    # Loading data
    feature_dim = 0
    path_first_model = conf.path_punctuation

    conf.remove_punctuation = False
    conf.use_features = False
    x, y, x_val, y_val, x_test, y_test_punct, vocab = load_and_preprocess(conf, padding=True)
    test_dataset_punctuation = PytorchDataset(x_test, y_test_punct)
    test_dataloader_punctuation = DataLoader(test_dataset_punctuation, batch_size=conf.batch_size, num_workers=2)

    if conf.test_mode == "features":
        conf.use_features = True
        x, y, x_val, y_val, x_test, y_test, vocab, data, data_v, data_t = load_and_preprocess(conf, padding=True)
        features = np.array([feature for feature in data]).transpose()
        features_test = np.array([feature for feature in data_t]).transpose()
        feature_dim = features_test.shape[1]
        test_dataset_features = PytorchFeatureDataset(x_test, features_test, y_test)
        test_dataloader = DataLoader(test_dataset_features, batch_size=conf.batch_size, num_workers=2)

        path_second_model = conf.path_punctuation_features
    else:
        conf.use_features = False
        conf.remove_punctuation = True
        x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
        test_dataset = PytorchDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

        path_second_model = conf.path_no_punctuation
    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model without punctuation
    first_model = torch.load(path_first_model)
    first_model = first_model.to(device)
    second_model = torch.load(path_second_model)
    second_model = second_model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Conduct test
    table = mcnemar.evaluate_two_models(first_model, second_model, test_dataloader_punctuation, test_dataloader,
                                        device, criterion, y_test_punct, y_test, num_labels=2, features=conf.use_features,
                                        batch_size=conf.batch_size)

    mcnemar.make_tests(table, exact=True if (table[0, 1] + table[1, 0] < 25) else False, correction=conf.correction)



