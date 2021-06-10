from util.utils import load_and_preprocess, logger, calculate_statistics, train, evaluate, load_preprocess_one_example
from util.dataloader import PytorchDataset, PytorchFeatureDataset
from torch.utils.data import DataLoader
from podium.vectorizers import GloVe
from pathlib import Path
from shutil import copy

import torch.nn as nn
import numpy as np
import importlib
import torch


if __name__ == '__main__':
    # Define configuration path
    conf_path = Path("../saves/TAR_models/GRU_DNN/punct/config.py")

    # Get configuaration
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Set seeds for reproducibility
    seed = conf.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setting hyper-parameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing the best model
    model_no_punct = torch.load(conf.path_no_punctuation)
    model_no_punct.to(device)
    model_no_punct.eval()

    model_punct = torch.load(conf.path_punctuation)
    model_punct.to(device)
    model_punct.eval()

    model_punct_features = torch.load(conf.path_punctuation_features)
    model_punct_features.to(device)
    model_punct_features.eval()

    # getting the input
    examples = ["The most exciting way to start a Friday: a presentation on structurally deficient bridges. #sarcasm #lobbyinglife",
                "Someone needs to stop me before I kill someone 😡 love waking up in the worst fcking mood #Not",
                "Joe Cole just casually walking past...",
                "@dsobek No ... now someone will start a rumor of $ABBV pricing their HCV drug same or higher than $GILD #sarcasm"]
    for example in examples:
        #example = input("Please enter the input text >> ")
        print(example)
        raw_text = example.strip("\n")

        x_np, x_p, x_pf, f_pf = load_preprocess_one_example(conf, raw_text, device)
        features_pf = torch.tensor(np.array([feature for feature in f_pf]).transpose(), dtype=torch.long).to(device)

        output_no_punct = model_no_punct(x_np, [len(x_np)])
        output_punct = model_punct(x_p, [len(x_p)])
        output_punct_features = model_punct_features(x_pf, [len(x_pf)], features_pf)

        probs_no_punct = torch.sigmoid(output_no_punct)
        probs_punct = torch.sigmoid(output_punct)
        probs_punct_features = torch.sigmoid(output_punct_features)

        _, class_no_punct = torch.max(probs_no_punct, dim=1)
        _, class_punct = torch.max(probs_punct, dim=1)
        _, class_punct_features = torch.max(probs_punct_features, dim=1)

        print(f"Prediction from model 1: {100*probs_no_punct[class_no_punct].detach().cpu().numpy()[0][0]:.2f}% for class: "
              f"{'Sarcasm' if class_no_punct == 1 else 'No sarcasm'}")

        print(f"Prediction from model 2: {100*probs_punct[class_punct].detach().cpu().numpy()[0][0]:.2f}% for class: "
              f"{'Sarcasm' if class_punct == 1 else 'No sarcasm'}")

        print(f"Prediction from model 3: {100*probs_punct_features[class_punct_features].detach().cpu().numpy()[0][0]:.2f}% for class: "
              f"{'Sarcasm' if class_punct_features == 1 else 'No sarcasm'}")
