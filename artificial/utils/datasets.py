import csv

import numpy as np


def make_spiral(n_samples=1000, n_components=2, n_classes=2,
                random_state=None):
    """Spiral Data Set.

    Generates the artificial data set Spiral with the parameters passed.

    Parameters
    ----------
    n_samples: int, default=1000
        The  number of samples of the data set.

    n_components: int, default=2
        The number of features. That is, the dimensional space in which the
        data set is embedded.

    n_classes: int, default=2
        The number of classes assumed by the samples.

    random_state: RandomState, default=None
        The random state object used for debugging purposes to control
        randomness.

    References
    ----------
    [1] Spiral Data set.
        "Convolutional Neural Networks for Visual Recognition", [Online],
        available at: <http://cs231n.github.io/neural-networks-case-study/>
    """

    random_state = random_state or np.random.RandomState()

    n_group_size = n_samples // n_classes

    X = np.zeros((n_samples, n_components))
    y = np.zeros(n_samples, dtype='uint8')

    for j in range(n_classes):
        indices = range(n_group_size * j, n_group_size * (j + 1))

        r = np.linspace(0.0, 1, n_group_size)
        t = (np.linspace(j * 4, (j + 1) * 4, n_group_size) +
             random_state.randn(n_group_size) * 0.2)

        X[indices] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[indices] = j

    return X, y


def iris(n_features=4):
    with open('data/iris.csv', 'rt', encoding='utf-8') as csv_file:
        lines = csv.reader(csv_file, delimiter=',')

        X = np.array(list(lines))

        y = X[:, -1]
        X = X[:, :n_features].astype(float)

        attributes = dict(zip(set(y), range(y.shape[0])))
        y = np.array([attributes[a] for a in y])

    return X, y
