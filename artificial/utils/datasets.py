import numpy as np


def make_vortex(n_samples=1000, n_components=2, n_classes=2,
                random_state=None):
    """Make vortex dataset."""

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
