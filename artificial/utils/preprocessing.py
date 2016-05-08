"""Pre-processing Helpers for Data Sets."""


# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

def scale(X):
    """Center and scale data set's features."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1

    return (X - mean) / std
