from unittest import TestCase

from artificial import agents, base
from artificial.searches import adversarial


class _TestAdversarialEnv(base.Environment):
    def update(self):
        for agent in self.agents:
            action = agent.act()

            self.current_state = _TestState(
                self.current_state.data + 1 if action == 1 else -1
            )

    def generate_random_state(self):
        return _TestState(self.random_state.randint(-10, 10))


class _TestState(base.State):
    pass


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        return [_TestState(state.data - 1, action=0),
                _TestState(state.data + 1, action=1)]

    def utility(self, state):
        return abs(state.data - 10)


class RandomTest(TestCase):
    def setUp(self):
        self.env = _TestAdversarialEnv(_TestState(0))
        self.agent = _UtilityTestAgent(search=adversarial.Random,
                                       environment=self.env,
                                       actions=None)

    def test_sanity(self):
        s = adversarial.Random(agent=self.agent, root=self.env.current_state)
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertGreater(len(actions), 0)
