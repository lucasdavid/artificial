import numpy as np
from unittest import TestCase
from numpy import testing

from artificial.utils import preprocessing


class PreprocessingTest(TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(0)

    def test_scale(self):
        # Expected centroid to be zero.
        expected = np.zeros(4)

        X = self.random_state.rand(10, 4)
        scaled_X = preprocessing.scale(X)

        actual = scaled_X.sum(axis=0)

        testing.assert_array_almost_equal(actual, expected)
