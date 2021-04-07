import pandas as pd

encoding = 'UTF-8'
train_path = '../datasets/train/SemEval2018-T3-train-task'
test_path = '../datasets/goldtest_Task'
text_header = 'Tweet text'
label_header = 'Label'
test_file_prefix = 'SemEval2018-T3_gold_test_task'


def load_train_data(task: str, emojis: bool = True, irony_hashtags: bool = False):
    """
    Loads SemEVAL2018 Task 3 training datasets (tweets and labels), depending on specification.

    :param task: The task for which the training data is being loaded, must be 'A' or 'B'. String.
    :param emojis: Determines if loaded dataset contains emojis. Boolean.
    :param irony_hashtags: Determines if loaded dataset contains hashtags regarding irony,
    emojis must be True as no dataset which contains hashtags and not emojis exists. Boolean.
    :return: Training data -- Pandas dataframes x and y containing tweets and their labels, respectively.
    """

    if task not in ['A', 'B']:
        raise RuntimeError(
            f'Task {task} does not exist, task value must be either \'A\' or \'B\'.')

    if not emojis and irony_hashtags:
        raise RuntimeError(
            'There is no dataset which contains hashtags but not emojis,'
            'set emojis to True if you want to use hashtags.')

    dataset_path = train_path + task
    dataset_path += '_emoji' if emojis else ''
    dataset_path += '_hashtags' if irony_hashtags else ''
    dataset_path += '.txt'

    df = pd.read_csv(dataset_path, delimiter='\t', header=0, encoding=encoding)
    x = df[text_header]
    y = df[label_header]

    return x, y


def load_test_data(task: str, emojis: bool = True):
    """
    Loads SemEVAL2018 Task 3 test datasets (tweets and labels), depending on specification.

    :param task: The task for which the training data is being loaded, must be 'A' or 'B'. String.
    :param emojis: Determines if loaded dataset contains emojis. Boolean.
    :return: Test data -- Pandas dataframes x and y containing tweets and their labels, respectively.
    """

    if task not in ['A', 'B']:
        raise RuntimeError(
            f'Task {task} does not exist, task value must be either \'A\' or \'B\'.')

    dataset_path = test_path + task + '/' + test_file_prefix + task
    dataset_path += '_emoji' if emojis else ''
    dataset_path += '.txt'

    df = pd.read_csv(dataset_path, delimiter='\t', header=0, encoding=encoding)
    x = df[text_header]
    y = df[label_header]

    return x, y


# test
if __name__ == '__main__':
    test_task = 'B'
    test_emojis = True
    test_irony_hashtags = False

    print(f'Loading training dataset: '
          f'task = {test_task}, emojis = {test_emojis}, irony_hashtags = {test_irony_hashtags}')
    x, y = load_train_data(test_task, emojis=test_emojis, irony_hashtags=test_irony_hashtags)
    print('Loaded labels:', y)
    print('Loaded tweets:', x)

    print(f'Loading test dataset: '
          f'task = {test_task}, emojis = {test_emojis}')
    x, y = load_test_data(test_task, emojis=test_emojis)
    print('Loaded labels:', y)
    print('Loaded tweets:', x)
