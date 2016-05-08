import numpy as np


def train_test_split(X, y, train_size, random_state=None):
    """Split a data set into two different subsets."""
    n_samples = X.shape[0]
    train_size = int(train_size * n_samples)

    if random_state is None:
        random_state = np.random.RandomState()

    p = random_state.permutation(n_samples)
    X, y = X[p], y[p]
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

