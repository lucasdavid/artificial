import random
from unittest import TestCase

from artificial import base, agents
from artificial.base import State


class _TState(State):
    @property
    def is_goal(self):
        return self.h() == 0

    def h(self):
        return abs(self.data - 10)


class _TEnv(base.Environment):
    def update(self):
        pass


class TableDrivenAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState(0), random_generator=random.Random(0))

    def test_sanity(self):
        action_map = {
            hash(0): 1,
            hash(1): 10,
            hash(10): 11,
            hash(11): 100,
            hash(100): 101,
        }

        tda = agents.TableDrivenAgent(action_map, self.env,
                                      action_map.values(),
                                      verbose=True)

        self.assertIsNotNone(tda)

    def test_perceive(self):
        action_map = {
            hash(0): 1,
            hash(1): 10,
            hash(10): 11,
            hash(11): 100,
            hash(100): 101,
        }

        tda = agents.TableDrivenAgent(action_map, self.env,
                                      action_map.values(),
                                      verbose=True)

        tda.perceive()
        self.assertEqual(tda.percepts, str(hash(_TState(0))))

        self.env.current_state = _TState(100)
        tda.perceive()
        self.assertEqual(tda.percepts,
                         str(hash(_TState(0))) + str(hash(_TState(100))))

    def test_act(self):
        action_map = {
            str(hash(0)): 1,
            str(hash(1)): 10,
            str(hash(10)): 11,
            str(hash(11)): 100,
            str(hash(100)): 101,
        }

        tda = agents.TableDrivenAgent(action_map, self.env,
                                      action_map.values(),
                                      verbose=True)

        tda.perceive()
        self.assertEqual(tda.percepts, str(hash(_TState(0))))

        action = tda.act()
        self.assertEqual(action, 1)

        self.env.current_state = _TState(100)
        tda.perceive()
        self.assertEqual(tda.percepts,
                         str(hash(_TState(0))) + str(hash(_TState(100))))

        # State "hash(0)+hash(100)" is not described on table.
        tda.perceive()
        action = tda.act()
        self.assertIsNone(action)

        # State is not described on table and warning is not raised.
        tda.verbose = False
        action = tda.act()
        self.assertIsNone(action)


class SimpleReflexAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState(0))

    def test_sanity(self):
        rules = {_TState(0): 1, _TState(1): 2, _TState(2): 1}

        sra = agents.SimpleReflexAgent(rules, self.env,
                                       rules.values(),
                                       verbose=True)

        self.assertIsNotNone(sra)

    def test_act(self):
        rules = {_TState(0): 1, _TState(1): 2, _TState(2): 1}

        sra = agents.SimpleReflexAgent(rules, self.env,
                                       rules.values(),
                                       verbose=True)

        action = sra.perceive().act()
        self.assertEqual(action, 1)

        self.env.current_state = _TState(3)
        action = sra.perceive().act()
        self.assertIsNone(action)

        sra.verbose = False
        action = sra.perceive().act()
        self.assertIsNone(action)


class _TestModelBasedAgent(agents.ModelBasedAgent,
                           agents.SimpleReflexAgent):
    def predict(self, state):
        action = self.rules[state] if state in self.rules else None
        children = []

        if action:
            # An state has a single action associated =>
            # takes to a single state. Although this guy is
            # very limited, this is just a test.
            children.append(_TState(action, action=action))

        return children


class ModelBasedAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState(0))

    def test_sanity(self):
        rules = {_TState(0): 1, _TState(1): 2, _TState(2): 1}
        sra = _TestModelBasedAgent(rules, self.env,
                                   rules.values(),
                                   verbose=True)

        action = sra.perceive().act()
        self.assertEqual(sra.last_state, _TState(0))
        self.assertEqual(sra.last_action, 1)
        self.assertEqual(sra.last_action, action)

        # Suddenly, the environment becomes undefined!
        self.env.current_state = None
        sra.perceive()

        # The last state known is 0.
        self.assertEqual(sra.last_known_state, _TState(0))

        # The last state is a guest of what would 
        # happen if action 1 were taken.
        self.assertEqual(sra.last_state, _TState(1, action=1))
