"""State Test"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import random
from unittest import TestCase

from artificial.base import State

random_generator = random.Random(0)


class _S(State):
    @classmethod
    def random(cls):
        return cls(random_generator.randint(0, 4))

    @property
    def is_goal(self):
        return self.data == 2


class StateTest(TestCase):
    def test_sanity(self):
        s = _S(None)
        self.assertIsNotNone(s)

    def test_equals(self):
        s1, s2, s3 = (_S([1, 2, 3, 4]), _S([1, 2, 3, 4]),
                      _S([1, 3, 2, 1]))

        self.assertEqual(s1, s1)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

        s1, s2 = _S({1, 2, 3}), _S({1, 2, 3})
        self.assertEqual(s1, s2)

        s1, s2 = _S({1, 2, 3}), _S([1, 2, 3])
        self.assertNotEqual(s1, s2)

    def test_mitosis(self):
        s1 = _S([1, 2, 3, 4])
        s2 = s1.mitosis()

        self.assertEqual(s1, s2.parent)

        s2.data[0] = -1
        self.assertNotEqual(s1.data[0], s2.data[0])

        s2.data[0] = 1
        self.assertListEqual(s1.data, s2.data)

        s3 = s1.mitosis(g=1)
        self.assertEqual(s1, s3.parent)

    def test_h(self):
        s1 = _S([1, 2, 3, 4])
        self.assertEqual(s1.h(), 0)

    def test_f(self):
        expected = 298321
        s1 = _S([1, 2, 3, 4], g=expected)
        actual = s1.f()

        # f() = g() + h() <=> f() = g()
        self.assertEqual(actual, expected)

    def test_is_goal(self):
        s = State([1, 2, 3])

        self.assertFalse(s.is_goal)

        s1 = _S(0)
        self.assertFalse(s1.is_goal, s1)

        s1.data = 2
        self.assertTrue(s1.is_goal, s1)

        s2 = _S(2)
        self.assertTrue(s2.is_goal, s2)

    def test___hash__(self):
        actual, expected = (hash(_S([1, 2, 3, 4])),
                            hash(str([1, 2, 3, 4])))
        self.assertEqual(actual, expected)

    def test___str__(self):
        s1 = str(_S([10, 20, 1, 5]))
        self.assertTrue(s1)
