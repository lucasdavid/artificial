"""Simple Path Finding Example.

This example shows how the IterativeDeepening algorithm can be used by an
agent to find the best path between two cities in a simple graph.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import logging

import artificial as art

logger = logging.getLogger('artificial')
logger.setLevel(logging.DEBUG)


class CityState(art.base.State):
    """City State.

    The definition of one state (a city) of this problem.

    """

    @property
    def is_goal(self):
        return self.data == 3

    def __str__(self):
        return 'city: %d, g: %d' % (self.data, self.g)

    def __eq__(self, other):
        # A CityState is equal to another when they share the same city-code.
        return isinstance(other, CityState) and self.data == other.data


class SimplePathFinding(art.base.Environment):
    """Simple Path Finding Environment.

    The definition of our problem domain.

    """
    real_cost = 0
    d = [
        [0, 6, 12, 25],
        [6, 0, 5, 10],
        [12, 5, 0, 5],
        [25, 10, 5, 0],
    ]
    incidence = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]

    def update(self):
        for agent in self.agents:
            agent.perceive()
            next_city = agent.act()

            current = self.current_state.data

            if (next_city is None or current == next_city or
                    not self.incidence[current][next_city]):
                # This agent doesn't know what to do.
                continue

            self.current_state = CityState(next_city)
            self.real_cost += self.d[current][next_city]


class RoutePlanner(art.agents.UtilityBasedAgent):
    """RoutePlanner Agent.

    A planner that evaluates the incidence and distance graphs at
    every iteration and decide to which city they should move on next
    (this was defined in `GoalBasedAgent.act` method, which returns
     an action -- or city -- to perform).

    We are left with setting the specifics of our problem: to define
    how the Agent predicts the transitions in the graph will be. Luckily,
    our agent has all info it needs (that's a classic routing problem).

    """

    def predict(self, state):
        """Predict possible cities that can be reached from `state`.

        :param state: CityState instance, root consider for the prediction.
        :return: list of CityState, referencing the cities that can be reached
                 from `state`.
        """
        current_city = state.data

        return [CityState(data=city,
                          parent=state,
                          action=city,
                          g=state.g + self.environment.d[current_city][city])
                # For every possible city...
                for city in self.actions
                # Only neighbors are considered (cities connected by a road)
                if self.environment.incidence[current_city][city]]


def main():
    print(__doc__)

    env = SimplePathFinding(initial_state=CityState(0))

    env.agents += [
        RoutePlanner(environment=env,
                     # Uses Iterative-deepening search.
                     search=art.searches.fringe.IterativeDeepening,
                     # Depths searched are 2 and 3.
                     search_params={'iterations': range(2, 4)},
                     # Can move to any city.
                     actions=list(enumerate(SimplePathFinding.g)))
    ]

    i, max_iterations = 0, 14

    print('Initial state: {%s}\n' % str(env.current_state))

    try:
        while i < max_iterations and not env.finished():
            i += 1
            print('#%i' % i)
            env.update()
            print('Current state: {%s}\n' % str(env.current_state))

        print('Solution found! (cost: %.1f) :-)' % env.real_cost
              if env.current_state.is_goal
              else 'Solution not found. :-(')

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
