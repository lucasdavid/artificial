"""Environment Test"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

from unittest import TestCase

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock

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

        built = env.build()
        self.assertEqual(env, built)

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

        env.current_state = _S(2)
        self.assertTrue(env.finished())

    def test_live(self):
        env = _E(_S(140202))
        env.build = MagicMock(side_effect=lambda: env)
        env.update = MagicMock(side_effect=lambda: env)

        expected_cycles = 5
        env.live(n_cycles=expected_cycles)
        env.build.assert_any_call()
        env.update.assert_any_call()

        env.live(n_cycles=expected_cycles)
        env.build.assert_any_call()
        env.update.assert_any_call()

        env.update = MagicMock(side_effect=KeyboardInterrupt)

        env.live(n_cycles=expected_cycles)
        env.build.assert_any_call()
        env.update.assert_any_call()

        env.live(n_cycles=expected_cycles)
        env.build.assert_any_call()
        env.update.assert_any_call()

    def test_current(self):
        if _E.current():
            _E.current().dispose()

        with self.assertRaises(RuntimeError):
            _E.current()

        env = _E()
        self.assertEqual(env, _E.current())

        env1 = _E()
        self.assertEqual(env1, _E.current())
        env2 = _E()
        self.assertNotEqual(env1, _E.current())
        self.assertEqual(env2, _E.current())

        del env1, env2
