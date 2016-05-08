from unittest import TestCase

import warnings
from artificial import base, agents
from artificial.base import State
from artificial.searches import fringe


class _S(State):
    @property
    def is_goal(self):
        return self.data == 10

    def h(self):
        return abs(self.data - 10)


class _TEnv(base.Environment):
    def update(self):
        pass


class TableDrivenAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_S(0))

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
        self.assertEqual(tda.percepts, str(hash(_S(0))))

        self.env.current_state = _S(100)
        tda.perceive()
        self.assertEqual(tda.percepts,
                         str(hash(_S(0))) + str(hash(_S(100))))

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
        self.assertEqual(tda.percepts, str(hash(_S(0))))

        action = tda.act()
        self.assertEqual(action, 1)

        self.env.current_state = _S(100)
        tda.perceive()
        self.assertEqual(tda.percepts,
                         str(hash(_S(0))) + str(hash(_S(100))))

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
        self.env = _TEnv(_S(0))

    def test_sanity(self):
        rules = {_S(0): 1, _S(1): 2, _S(2): 1}

        sra = agents.SimpleReflexAgent(rules, self.env,
                                       rules.values(),
                                       verbose=True)

        self.assertIsNotNone(sra)

    def test_act(self):
        rules = {_S(0): 1, _S(1): 2, _S(2): 1}

        sra = agents.SimpleReflexAgent(rules, self.env,
                                       rules.values(),
                                       verbose=True)

        action = sra.perceive().act()
        self.assertEqual(action, 1)

        self.env.current_state = _S(3)
        action = sra.perceive().act()
        self.assertIsNone(action)

        sra.verbose = False
        action = sra.perceive().act()
        self.assertIsNone(action)


class ModelBasedAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_S(0))

    def test_infer_state(self):
        class _TestModelBasedAgent(agents.ModelBasedAgent,
                                   agents.SimpleReflexAgent):
            def predict(self, state):
                a = self.rules[state] if state in self.rules else None
                children = []

                if a:
                    # An state has a single action associated =>
                    # takes to a single state. Although this guy is
                    # very limited, this is just a test.
                    children.append(_S(a, action=a))

                return children

        rules = {_S(0): 1, _S(1): 2, _S(2): 1}
        sra = _TestModelBasedAgent(rules, self.env,
                                   rules.values(),
                                   verbose=True)

        action = sra.perceive().act()
        self.assertEqual(sra.last_state, _S(0))
        self.assertEqual(sra.last_action, 1)
        self.assertEqual(sra.last_action, action)

        # Suddenly, the environment becomes undefined!
        self.env.current_state = None
        sra.perceive()

        # The last state known is 0.
        self.assertEqual(sra.last_known_state, _S(0))

        # The last state is a guest of what would 
        # happen if action 1 were taken.
        self.assertEqual(sra.last_state, _S(1, action=1))

    def test_undefined_action_warning(self):
        class _TestModelBasedAgent(agents.ModelBasedAgent,
                                   agents.SimpleReflexAgent):
            def predict(self, state):
                a = self.rules[state] if state in self.rules else None
                children = []

                if a:
                    children.append(_S(a))

                return children

        rules = {_S(0): 1, _S(1): 2, _S(2): 1}
        sra = _TestModelBasedAgent(rules, self.env,
                                   rules.values(),
                                   verbose=True)
        sra.perceive().act()

        # Suddenly, the environment becomes undefined!
        self.env.current_state = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sra.perceive()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))


class _TestGoalBasedAgent(agents.GoalBasedAgent):
    def predict(self, state):
        return [
            _S(state.data - 1, action=0, parent=state),
            _S(state.data + 1, action=1, parent=state),
            _S(state.data, action=2, parent=state),
        ]


class GoalBasedAgentTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_S(0))

    def test_sanity(self):
        gba = _TestGoalBasedAgent(fringe.BreadthFirst,
                                  environment=self.env,
                                  actions=[0, 1, 2])
        self.assertIsNotNone(gba)

    def test_act(self):
        gba = _TestGoalBasedAgent(fringe.BreadthFirst,
                                  environment=self.env,
                                  actions=[0, 1, 2])
        for _ in range(10):
            self.assertEqual(gba.perceive().act(), 1)

        # Test verbosity.
        gba = _TestGoalBasedAgent(fringe.BreadthFirst,
                                  verbose=True,
                                  environment=self.env,
                                  actions=[0, 1, 2])
        for _ in range(10):
            self.assertEqual(gba.perceive().act(), 1)

    def test_is_agent_goal(self):
        gba = _TestGoalBasedAgent(fringe.BreadthFirst,
                                  environment=self.env,
                                  actions=[0, 1, 2])

        self.assertFalse(gba.is_goal(_S(0)))
        self.assertTrue(gba.is_goal(_S(10)))
