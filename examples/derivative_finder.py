"""Derivative Finder Example.

This example demonstrates how a artificially intelligent agent can use
`Hill Climbing` to find the domain point of a function `f` in which this
functions' derivative is zero.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016


"""

import random
import time

import artificial as art


class DerivativeState(art.base.State):
    @property
    def is_goal(self):
        return abs(self.h()) <= DifferentialEnvironment.max_error

    def h(self):
        f, d, x = (DifferentialEnvironment.f,
                   DifferentialEnvironment.delta,
                   self.data)

        return (f(x + d) - f(x - d)) / (2 * d)

    def __str__(self):
        f = DifferentialEnvironment.f
        return 'f(%f)=%f, dy/dx = %f' % (self.data, f(self.data), self.h())

    @classmethod
    def random(cls):
        return cls(data=random.random() * 100 - 50)


class DifferentialEnvironment(art.base.Environment):
    max_error = .01
    delta = .0008
    failed = False

    actions = (
        {'label': 'move-right'},
        {'label': 'move-left'},
    )

    state_class_ = DerivativeState

    @staticmethod
    def f(x):
        """The function being differentiated.

        This is just a toy! In a real application, we might also not know this
        function, and only have evaluations for specific domain points
        (ordinary or partial differential equations).

        """
        return -(x - 10) ** 5 + x ** 3

    def update(self):
        for agent in self.agents:
            agent.perceive()
            x = agent.act()

            if x is None:
                # When hill climber only fails if it's
                # already on the top of a local maximum.
                self.failed = True
                break

            self.current_state = DerivativeState(x)

    def finished(self):
        return self.failed or self.current_state.is_goal


class DerivativeFinder(art.agents.UtilityBasedAgent):
    def act(self):
        return (self.search
                .restart(root=self.last_known_state)
                .search()
                .solution_candidate_
                .data)

    def predict(self, state):
        x, d = state.data, self.environment.delta
        return [DerivativeState(x - d), DerivativeState(x + d)]

    def utility(self, state):
        return -abs(state.h())


def main():
    print(__doc__)

    env = DifferentialEnvironment(initial_state=DerivativeState(0))
    env.agents += [
        DerivativeFinder(environment=env,
                         search=art.searches.local.HillClimbing,
                         search_params=dict(restart_limit=10),
                         actions=(0, 1))]

    print('initial solution candidate:  {%s}' % str(env.current_state))

    env.build()

    start = time.time()

    try:
        # Just a single update, not inside a while loop.
        env.update()
        print('best solution candidate found: {%s}' % str(env.current_state))
    except KeyboardInterrupt:
        print('canceled')
    finally:
        print('\nTime elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
