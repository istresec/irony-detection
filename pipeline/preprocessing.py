from podium import TabularDataset, Dataset, Field, Vocab, LabelField, Iterator
import pandas as pd
from util import dataloader


def lowercase(raw):
    return raw.lower()


def preprocess_and_tokenize(dataset: pd.DataFrame, text_name: str = 'text', label_name: str = 'label') -> Dataset:
    """
    Preprocesses text data and returns a Podium dataset.

    :param dataset: Dataset to be preprocessed and tokenized, containing text and labels. Pandas DataFrame.
    :param text_name: The name of the text column in the dataset Pandas DataFrame, 'text' by default. String.
    :param label_name: The name of the label column in the dataset Pandas DataFrame, 'label by default. String.
    :return: A finalized Podium dataset, preprocessed and tokenized.
    """
    text = Field(name='input_text',
                 tokenizer="split",
                 numericalizer=Vocab(),
                 keep_raw=False,
                 pretokenize_hooks=[lowercase],
                 posttokenize_hooks=[])
    label = LabelField(name='target', is_target=True)
    fields = {text_name: text, label_name: label}

    dataset = TabularDataset.from_pandas(df=dataset, fields=fields)
    dataset.finalize_fields()

    return dataset


# test
if __name__ == '__main__':
    test_task = 'A'
    test_type = 'train'

    train_data = dataloader.load_train_data(test_task)

    train_dataset = preprocess_and_tokenize(train_data)
    print(train_dataset)

    # play with batch iteration
    dataset_iter = Iterator(train_dataset, batch_size=32)
    print('Iterating through dataset minibatches:')
    for batch_num, batch in enumerate(dataset_iter):
        if batch_num % 20 == 0:
            print(f'\tBatch {batch_num}: {hash(str(batch))} (hashed string of the batch)')
