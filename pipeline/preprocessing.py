from podium import TabularDataset, Dataset, Field, Vocab, LabelField, Iterator
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from podium.preproc import TextCleanUp

from pipeline.irony_detection_preprocessor import IronyDetectionPreprocessor
from util import dataloader

import pandas as pd
import numpy as np


def lower(raw: str) -> str:
    """
    Returns a lowercase copy of the input text.

    :param raw: Text which will be turned to lowercase. String.
    :return: A lowercase copy of the input text. String.
    """
    return raw.lower()


def preprocess_and_tokenize(dataset: pd.DataFrame, text_name: str = 'text', label_name: str = 'label',
                            finalize: bool = True,
                            remove_punct: bool = True) -> Dataset:
    """
    Preprocesses text data and returns a Podium dataset.

    :param dataset: Dataset to be preprocessed and tokenized, containing text and labels. Pandas DataFrame.
    :param text_name: The name of the text column in the dataset Pandas DataFrame, 'text' by default. String.
    :param label_name: The name of the label column in the dataset Pandas DataFrame, 'label by default. String.
    :param finalize: Determines if dataset is returned finalized or not, True by default.
    :param remove_punct: Determines if punctuation is removed or not. Boolean.
    :return: A Podium Dataset, preprocessed and tokenized, and a Podium Vocab.
    """
    vocab = Vocab()

    cleanup = IronyDetectionPreprocessor(remove_punct=remove_punct)
    text = Field(name='input_text',
                 tokenizer=TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize,
                 numericalizer=vocab,
                 keep_raw=False,
                 pretokenize_hooks=[lower, cleanup],
                 posttokenize_hooks=[])
    label = LabelField(name='target', is_target=True)
    fields = {text_name: text, label_name: label}

    dataset = TabularDataset.from_pandas(df=dataset, fields=fields)

    if finalize:
        dataset.finalize_fields()

    return dataset, vocab


def tf_idf_vectorization(dataset: Dataset, max_features: int = 15000, remove_punct: bool = True, vocabulary=None):
    """
    Vectorizes each instance in the dataset and returns the TF-IDF vectorization.

    :param dataset: Dataset to be vectorized, containing input_text and labels. Podium.Dataset.
    :param max_features: max number of features in the vocabulary. Integer.
    :param remove_punct: flag for removing or keeping the punctuations. Boolean.
    :param vocabulary: vocabulary.
    :return: TF-IDF vectorization of given dataset and constructed vocabulary
    """
    tf_idf_vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True,
                                        preprocessor=IronyDetectionPreprocessor(remove_punct=remove_punct),
                                        tokenizer=TweetTokenizer(preserve_case=False, reduce_len=True,
                                                                 strip_handles=True).tokenize, vocabulary=vocabulary)

    x = tf_idf_vectorizer.fit_transform(dataset)
    vocab = tf_idf_vectorizer.vocabulary_

    """
    # Podium code for TfIdf vectorization
    tf_idf = TfIdfVectorizer()
    tf_idf.fit(dataset, field=dataset.field("input_text"))
    return tf_idf.transform(x)
    """
    return x, vocab


# test
if __name__ == '__main__':
    test_task = 'A'
    test_type = 'train'

    train_data = dataloader.load_train_data(test_task)
    test_data = dataloader.load_test_data(test_task)

    # TF-IDF
    tfidf_batch, vocab = tf_idf_vectorization(train_data["text"])
    tfidf_batch_t, _ = tf_idf_vectorization(test_data["text"], vocabulary=vocab)

    print(f"Shapes of TF-IDF batches: train = {tfidf_batch.shape}, test = {tfidf_batch_t.shape}")
    print(f"TF-IDF vectorization of first batch example:\n{tfidf_batch[0]}")  # vector representation
    print(f"TF-IDF vectorization of first batch example:\n{tfidf_batch_t[0]}")  # vector representation

    # podium word embedding preprocessing
    train_dataset, vocab = preprocess_and_tokenize(train_data, finalize=False, remove_punct=True)
    test_dataset, _ = preprocess_and_tokenize(test_data, finalize=False, remove_punct=True)

    train_dataset.finalize_fields(train_dataset, test_dataset)
    test_dataset.finalize_fields()

    x, y = train_dataset.batch(add_padding=True)
    x_t_p, y_t = test_dataset.batch(add_padding=True)

    x_t = np.ones((x_t_p.shape[0], x.shape[1]), dtype=np.integer)
    x_t[:, :x_t_p.shape[1]] = x_t_p

    print(f"Shapes | x: {x.shape}, y: {y.shape}")
    print(f"Shapes | x_t: {x_t.shape}, y_t: {y_t.shape}")

    print(f"First example of train data:\n{train_dataset[0]}")  # input text and label
    print(f"Batch of first example of input data:\n{x[0]}")  # dictionary indices

    print(f"First example of test data:\n{test_dataset[0]}")  # input text and label
    print(f"Batch of first example of test data:\n{x_t[0]}")  # dictionary indices
