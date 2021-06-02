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

    # Initialize GloVe
    glove = GloVe(dim=conf.glove_dim)

    # Loading data
    feature_dim = 0
    if conf.test_mode == "features":
        x, y, x_val, y_val, x_test, y_test, vocab, data, data_v, data_t = load_and_preprocess(conf, padding=True)
        features_test = np.array([feature for feature in data_t]).transpose()

        feature_dim = features_test.shape[1]
        test_dataset_features = PytorchFeatureDataset(x_test, features_test, y_test)
        test_dataloader = DataLoader(test_dataset_features, batch_size=conf.batch_size, num_workers=2)

        test_dataset = PytorchDataset(x_test, y_test)
        test_dataloader_punctuation = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)
    else:
        conf.remove_punctuation = True
        x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
        test_dataset = PytorchDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=2)

        conf.remove_punctuation = False
        x, y, x_val, y_val, x_test, y_test, vocab = load_and_preprocess(conf, padding=True)
        test_dataset_punctuation = PytorchDataset(x_test, y_test)
        test_dataloader_punctuation = DataLoader(test_dataset_punctuation, batch_size=conf.batch_size, num_workers=2)

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model without punctuation
    first_model = torch.load(conf.path_first_model)
    first_model = first_model.to(device)
    second_model = torch.load(conf.path_second_model)
    second_model = second_model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Conduct test
    table = mcnemar.evaluate_two_models(first_model, second_model, test_dataloader_punctuation, test_dataloader,
                                        device, criterion, num_labels=2, features=conf.use_features)

    mcnemar.make_tests(table, exact=conf.exact, correction=conf.correction)



