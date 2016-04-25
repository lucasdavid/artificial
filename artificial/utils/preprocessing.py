"""Pre-processing Helpers for Data Sets."""


# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

def scale(X):
    """Center and scale data set's features."""
    return (X - X.mean(axis=0)) / X.std(axis=0)
