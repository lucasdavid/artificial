from unittest import TestCase

from artificial.base import State, Environment


class _S(State):
    """A nice state where the goal is to reach `2`"""
    @property
    def is_goal(self):
        return self.data == 2

    def h(self):
        return abs(self.data - 2)


class _E(Environment):
    def update(self):
        pass


class EnvironmentTest(TestCase):
    def test_sanity(self):
        env = _E(_S(10))
        self.assertIsNotNone(env)
        self.assertEqual(env.current_state, _S(10))

    def test_add_agents(self):
        env = _E(None)
        expected = ['A', 'B', 'C']
        env.agents += expected

        self.assertListEqual(env.agents, expected)

    def test_finished(self):
        s = _S(0)
        env = _E(s)

        s.data = 1
        self.assertFalse(env.finished())

        s.data = 2
        self.assertTrue(env.finished())
