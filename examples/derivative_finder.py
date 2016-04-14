import random
import time

from artificial import base, agents
from artificial.searches.local import HillClimbing


class DerivativeState(base.State):
    @property
    def is_goal(self):
        return abs(self.h()) <= FunctionsEnvironment.max_error
    
    def h(self):
        f, d, x = (FunctionsEnvironment.actual_f,
                   FunctionsEnvironment.delta,
                   self.data)

        return (f(x + d) - f(x - d)) / (2 * d)

    def __str__(self):
        f = FunctionsEnvironment.actual_f
        return 'f(%.2f)=%.2f, dy/dx = %.2f' % (self.data, f(self.data),
                                               self.h())


class FunctionsEnvironment(base.Environment):
    max_error = .1
    delta = .0008
    failed = False

    actions = (
        {'label': 'move-right'},
        {'label': 'move-left'},
    )

    @classmethod
    def generate_random_state(cls):
        return DerivativeState(data=random.random() * 100 - 50)

    @staticmethod
    def actual_f(x):
        return -(x - 10) ** 5 + x ** 3

    def update(self):
        for agent in self.agents:
            agent.perceive()
            x = agent.act()

            if x is None:
                # When hill climber only fails if it's already on the top.
                self.failed = True
                break

            self.current_state = DerivativeState(x)

    def finished(self):
        return self.failed or self.current_state.is_goal


class DerivativeFinder(agents.UtilityBasedAgent):
    def act(self):
        return (self.search
                .restart(root=self.last_known_state)
                .search()
                .solution_candidate
                .data)

    def predict(self, state):
        x, d = state.data, self.environment.delta
        return [DerivativeState(x - d), DerivativeState(x + d)]

    def utility(self, state):
        return -abs(state.h())


def main():
    print('================================')
    print('Polynomial Approximation Example')
    print('================================\n')

    env = FunctionsEnvironment(initial_state=DerivativeState(0))
    env.agents += [
        DerivativeFinder(environment=env,
                         search=HillClimbing,
                         search_params=dict(restart_limit=10),
                         actions=(0, 1))]

    print('Initial:  {%s}' % str(env.current_state))

    start = time.time()

    try:
        env.update()
        print('Solution: {%s}' % str(env.current_state))

    except KeyboardInterrupt:
        pass

    finally:
        print('\nTime elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
