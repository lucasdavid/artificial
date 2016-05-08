from unittest import TestCase

import numpy as np
from artificial.utils import model_selection
from nose_parameterized import parameterized


class ModelSelectionTest(TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(0)

    @parameterized.expand([
        ((10, 4), .8, np.random.RandomState(0)),
        ((20, 2), .5, None),
    ])
    def test_train_test_split(self, expected_shape, expected_train_size,
                              expected_random_state):
        X = self.random_state.rand(*expected_shape)
        y = self.random_state.randint(10, size=expected_shape[0])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y,
            train_size=expected_train_size, random_state=expected_random_state)

        expected_train_shape = (int(expected_train_size * expected_shape[0]),
                                expected_shape[1])

        expected_test_shape = (X.shape[0] - expected_train_shape[0],
                               expected_train_shape[1])

        self.assertEqual(X_train.shape, expected_train_shape)
        self.assertEqual(X_test.shape, expected_test_shape)

        self.assertEqual(y_train.shape, expected_train_shape[:1])
        self.assertEqual(y_test.shape, expected_test_shape[:1])
