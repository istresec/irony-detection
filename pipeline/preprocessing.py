from podium import TabularDataset, Dataset, Field, Vocab, LabelField, Iterator
from sklearn.feature_extraction.text import TfidfVectorizer
from podium.vectorizers import TfIdfVectorizer
from nltk.tokenize import TweetTokenizer
from podium.preproc import TextCleanUp
from util import dataloader

import pandas as pd


def lower(raw: str) -> str:
    """
    Returns a lowercase copy of the input text.

    :param raw: Text which will be turned to lowercase. String.
    :return: A lowercase copy of the input text. String.
    """
    return raw.lower()


def preprocess_and_tokenize(dataset: pd.DataFrame, text_name: str = 'text', label_name: str = 'label',
                            remove_punct: bool = True) -> Dataset:
    """
    Preprocesses text data and returns a Podium dataset.

    :param dataset: Dataset to be preprocessed and tokenized, containing text and labels. Pandas DataFrame.
    :param text_name: The name of the text column in the dataset Pandas DataFrame, 'text' by default. String.
    :param label_name: The name of the label column in the dataset Pandas DataFrame, 'label by default. String.
    :param remove_punct: Determines if punctuation is removed or not. Boolean.
    :return: A finalized Podium dataset, preprocessed and tokenized.
    """
    cleanup = TextCleanUp(remove_punct=remove_punct)
    text = Field(name='input_text',
                 tokenizer=TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize,
                 numericalizer=Vocab(),
                 keep_raw=False,
                 pretokenize_hooks=[lower, cleanup],
                 posttokenize_hooks=[])
    label = LabelField(name='target', is_target=True)
    fields = {text_name: text, label_name: label}

    dataset = TabularDataset.from_pandas(df=dataset, fields=fields)
    dataset.finalize_fields()

    return dataset


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
                                        preprocessor=TextCleanUp(remove_punct=remove_punct),
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
    # train_dataset = preprocess_and_tokenize(train_data, vocab, remove_punct=True)
    # print(vocab)
    # test_dataset = preprocess_and_tokenize(test_data, vocab, remove_punct=True)
    # x, y = train_dataset.batch(add_padding=True)
    # x_t, y_t = test_dataset.batch(add_padding=True)
    tfidf_batch, vocab = tf_idf_vectorization(train_data["text"])
    print(vocab.shape)
    tfidf_batch_t, _ = tf_idf_vectorization(test_data["text"], vocabulary=vocab)

    # print(f"Shapes | x: {x.shape}, y: {y.shape}")
    # print(f"First example of input data:\n{train_dataset[0]}")  # input text and label
    # print(f"Batch of first example of input data:\n{x[0]}")  # dictionary indices
    print(f"Shapes of TF-IDF batches: train = {tfidf_batch.shape}, test = {tfidf_batch_t.shape}")
    print(f"TF-IDF vectorization of first batch example:\n{tfidf_batch[0]}")  # vector representation
    print(f"TF-IDF vectorization of first batch example:\n{tfidf_batch_t[0]}")  # vector representation

    """
    # play with batch iteration
    dataset_iter = Iterator(train_dataset, batch_size=32)
    print('Iterating through dataset minibatches:')
    for batch_num, batch in enumerate(dataset_iter):
        if batch_num % 20 == 0:
            print(f'\tBatch {batch_num}: {hash(str(batch))} (hashed string of the batch)')
    """
