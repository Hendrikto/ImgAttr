import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import scale


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'F1 score': f1_score(y_true, y_pred, average='weighted'),
        'Jaccard index': jaccard_score(y_true, y_pred, average='weighted'),
    }


def evaluate_classifier(classifier, X, y, n_splits=10):
    X = scale(X)

    splitter = StratifiedShuffleSplit(n_splits=n_splits)
    metrics = pd.DataFrame(columns=(
        'accuracy',
        'F1 score',
        'Jaccard index',
    ), index=range(n_splits), dtype=float)
    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        print(f'split {i + 1} of {n_splits}')
        classifier.fit(X[train_indices], y[train_indices])
        metrics.iloc[i] = compute_metrics(y[test_indices], classifier.predict(X[test_indices]))

    return metrics
