import time

from artificial import base, searches, agents


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
    delta = .008
    failed = False

    @staticmethod
    def actual_f(x):
        return -(x - 10) ** 5 + x ** 3

    def update(self):
        for agent in self.agents:
            agent.perceive()
            act = agent.act()

            if act is None:
                # When hill climber only fails if it's already on the top.
                self.failed = True
                break

            x = self.current_state.data

            if act == 0:
                self.current_state = DerivativeState(x + self.delta)

            elif act == 1:
                self.current_state = DerivativeState(x - self.delta)

    def finished(self):
        return self.failed or super().finished()


class DerivativeFinder(agents.UtilityBasedAgent):
    def predict(self, state):
        children = []

        s = state.mitosis(action=0)
        s.data += self.environment.delta
        children.append(s)

        s = state.mitosis(action=1)
        s.data -= self.environment.delta
        children.append(s)

        return children

    def utility(self, state):
        return -abs(state.h())


def main():
    print('==========================')
    print('Polynomial Approximation Example')
    print('==========================\n')

    i, max_iterations = 0, 1000

    env = FunctionsEnvironment(initial_state=DerivativeState(0))

    env.agents += [
        DerivativeFinder(environment=env,
                         search=searches.HillClimbing,
                         actions=(0, 1))]

    print('Initial state: {%s}\n' % str(env.current_state))

    start = time.time()

    try:
        while i < max_iterations and not env.finished():
            env.update()
            print('#%i current state: {%s}' % (i, str(env.current_state)))

            i += 1

    except KeyboardInterrupt:
        pass

    finally:
        print('\nTime elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
