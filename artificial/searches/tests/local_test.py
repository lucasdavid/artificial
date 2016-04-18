import random
from unittest import TestCase

from artificial import agents, base
from artificial.searches.local import HillClimbing, LocalBeam


random_generator = random.Random(0)


class _TState(base.State):
    @property
    def is_goal(self):
        return self.data == 100

    def h(self):
        return abs(self.data - 100)

    @classmethod
    def random(cls):
        return cls(random_generator.randint(-1000, 1000))


class _TestEnvironment(base.Environment):
    state_class_ = _TState
    
    def update(self):
        pass


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        children = [_TState(data=state.data + i - 5,
                            action=0 if i < 5 else 2 if i == 5 else 1)
                    for i in range(10)]

        return children


class HillClimbingTest(TestCase):
    def setUp(self):
        self.env = _TestEnvironment(_TState(0))
        self.agent = _UtilityTestAgent(HillClimbing, self.env, actions=None)

    def test_sanity(self):
        with self.assertRaises(AssertionError):
            HillClimbing(agent=None, root=_TState(10))

        s = HillClimbing(agent=self.agent, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        s = (HillClimbing(agent=self.agent)
             .restart(_TState(0))
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)

    def test_classic_strategy(self):
        s = (HillClimbing(agent=self.agent, strategy='classic')
             .restart(_TState(0))
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)

    def test_random_restart(self):
        s = (HillClimbing(agent=self.agent, restart_limit=2)
             .restart(_TState(0))
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)


class LocalBeamTest(TestCase):
    def setUp(self):
        self.env = _TestEnvironment(_TState(0),
                                    random_generator=random_generator)
        self.agent = _UtilityTestAgent(LocalBeam, self.env, actions=None)

    def test_sanity(self):
        with self.assertRaises(AssertionError):
            LocalBeam(agent=None, root=_TState(10), k=2)

        with self.assertRaises(ValueError):
            (LocalBeam(agent=self.agent, root=_TState(10), k='invalid')
             .search())

        s = LocalBeam(agent=self.agent, root=_TState(10), k=2)
        self.assertIsNotNone(s)

    def test_search(self):
        s = (LocalBeam(agent=self.agent, k=2, root=_TState(0))
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)

    def test_classic_strategy(self):
        s = (LocalBeam(agent=self.agent, root=self.env.current_state,
                       strategy='classic', k=2)
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)

    def test_random_restart(self):
        s = (LocalBeam(agent=self.agent, root=self.env.current_state,
                       k=2, restart_limit=2)
             .search())

        self.assertTrue(s.solution_candidate_.is_goal,
                        str(s.solution_candidate_))
        self.assertEqual(s.solution_candidate_.data, 100)

