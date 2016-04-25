

def scale(X):
    """Center and scale data set's features."""
    return (X - X.mean(axis=0)) / X.std(axis=0)
