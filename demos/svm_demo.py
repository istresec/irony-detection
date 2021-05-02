import argparse
import importlib
from pathlib import Path

from sklearn.metrics import accuracy_score

from benchmark_system.example import parse_dataset, featurize
from models.basic_model import BasicModel
from pipeline.preprocessing import preprocess_and_tokenize, tf_idf_vectorization
from util import dataloader

# Test SVM with basic model.
if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config file from which model and dataset is read")
    args = parser.parse_args()
    conf_path = Path(args.config)
    # This works too
    # conf_path = Path("..\configs\svm.py")

    print("Using Podium:")
    # Get configs
    spec = importlib.util.spec_from_file_location('module', conf_path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # Get data
    train_data = dataloader.load_train_data(conf.test_task, emojis=conf.test_emojis,
                                            irony_hashtags=conf.test_irony_hashtags)
    train_dataset = preprocess_and_tokenize(train_data, remove_punct=conf.remove_punctuation)
    x, y = train_dataset.batch(add_padding=True)
    tfidf_batch = tf_idf_vectorization(train_dataset, x)

    # Train model via tf-idf
    model = BasicModel(conf.backbone)
    model.fit(tfidf_batch, y.ravel())
    y_hat = model.predict(tfidf_batch)
    acc = accuracy_score(y_hat, y.ravel())
    print(f"Accuracy on the train set with tf-idf BOW: {acc:.4f}")

    # Train model via dictionary indices
    model = BasicModel(conf.backbone)
    model.fit(x, y.ravel())
    y_hat = model.predict(x)
    acc = accuracy_score(y_hat, y.ravel())
    print(f"Accuracy on the train set with dictionary indices: {acc:.4f}")

    print("\nUsing benchmark:")
    dataset_path = '../datasets/train/SemEval2018-T3-train-task' + conf.test_task
    dataset_path += '_emoji' if conf.test_emojis else ''
    dataset_path += '_ironyHashtags' if conf.test_irony_hashtags else ''
    dataset_path += '.txt'

    corpus, y =  parse_dataset(dataset_path)
    x = featurize(corpus)

    model.fit(x, y)
    y_hat = model.predict(x)
    acc = accuracy_score(y_hat, y)
    print(f"Accuracy on the train set with benchmark system (tf-idf BOW): {acc:.4f}")
