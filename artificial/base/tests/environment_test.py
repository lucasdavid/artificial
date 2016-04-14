from unittest import TestCase

from artificial.base import State, Environment


class _TestState(State):
    @property
    def is_goal(self):
        return self.data == 2


class _LocalTestEnvironment(Environment):
    def update(self):
        pass


class EnvironmentTest(TestCase):
    def test_sanity(self):
        env = _LocalTestEnvironment(None)
        self.assertIsNotNone(env)

    def test_add_agents(self):
        env = _LocalTestEnvironment(None)
        expected = ['A', 'B', 'C']
        env.agents += expected

        self.assertListEqual(env.agents, expected)

    def test_finished(self):
        s = _TestState(0)
        env = _LocalTestEnvironment(s)

        s.data = 1
        self.assertFalse(env.finished())

        s.data = 2
        self.assertTrue(env.finished())
