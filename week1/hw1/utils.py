from tqdm import tqdm_notebook as tqdm
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize


def evaluate_model(model, x_train, x_val, y_train, y_val):
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    print(f1_score(y_val, predictions, average='micro'))


def pred_for_sparce(clf, features, k=100):
    # sklearn doesn't work well with scipy sparce, so we work with slices
    pred = []
    n = features.shape[0]
    for i in tqdm(range(n // k + bool(n % k))):
        pred.append(clf.predict(features[i * k:(i + 1) * k].toarray()))
    pred = np.hstack(pred)[:n]
    return pred


def get_data(to_subm=False):
    train_data = pd.read_csv('./data/train.csv', index_col='id')
    test_data = pd.read_csv('./data/test.csv', index_col='id')
    y = train_data.values[:, 1].astype(int)
    x = train_data.values[:, 0]
    x_test = test_data.values[:, 0]

    x_tokens = np.array(json.load(open('./data/train_tokens.json', 'rb')))
    x_tokens_test = np.array(json.load(open('./data/test_tokens.json', 'rb')))

    train_idx, val_idx = train_test_split(np.arange(len(x)), train_size=0.8, random_state=0)
    if to_subm:
        train_idx = np.arange(len(x))  # for final submit

    return (x[train_idx], x[val_idx], x_test), \
           (x_tokens[train_idx], x_tokens[val_idx], x_tokens_test), \
           (y[train_idx], y[val_idx])


class WrapperClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper classifier helps select only certain columns for a pipeline
    """

    def __init__(self, clf, start_idx=0, stop_idx=-1):
        self.clf = clf
        self.start_idx = start_idx
        self.stop_idx = stop_idx

    def fit(self, X, y=None):
        self.clf.fit(normalize(X[:,self.start_idx:self.stop_idx]), y)
        return self

    def predict_proba(self, X, y=None):
        try:
            return self.clf.predict_proba(normalize(X[:, self.start_idx:self.stop_idx]))

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

    def predict(self, X, y=None):
        try:
            return self.clf.predict(normalize(X[:, self.start_idx:self.stop_idx]))

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
