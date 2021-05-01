from util.dataloader import load_train_data, load_test_data
from pipeline.preprocessing import preprocess_and_tokenize

import podium as pdm
import pandas as pd

task = 'A'
emojis = True
hashtags = False


def with_print(with_bool: bool):
    if with_bool is True:
        return 'with'
    else:
        return 'without'


if __name__ == '__main__':
    print(f'Testing pipeline for task {task}.\n')

    # load training data
    print(f'Loading training data for task {task}, '
          f'{with_print(emojis)} emojis and {with_print(hashtags)} irony hashtags.\n')
    train_data = load_train_data(task=task, emojis=emojis, irony_hashtags=hashtags)
    print(f'Training dataset loaded, the resulting pandas dataframe:\n{train_data}\n')

    # load test data
    print(f'Loading test dataset for task {task}, '
          f'{with_print(emojis)} emojis.\n')
    test_data = load_test_data(task=task, emojis=emojis)
    print(f'Test dataset loaded, the resulting pandas dataframe:\n{test_data}\n')

    # preprocess training data
    print('Preprocessing training data.\n')
    train_dataset = preprocess_and_tokenize(train_data)
    print(f'Preprocessed training data into a Podium dataset:\n{train_dataset}\n')

    # preprocess test data
    print('Preprocessing test data.\n')
    test_dataset = preprocess_and_tokenize(test_data)
    print(f'Preprocessed test data into a Podium dataset:\n{test_dataset}\n')
