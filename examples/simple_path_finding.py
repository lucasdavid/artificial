"""Simple Path Finding Example.

This example shows how the IterativeDeepening algorithm can be used by an
agent to find the best path between two cities in a simple graph.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import logging

import artificial as art
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('artificial')


class CityState(art.base.State):
    """City State.

    The definition of one state (a city) of this problem.

    """

    @property
    def is_goal(self):
        return self.data == 3

    def __str__(self):
        return 'city: %d, g: %d' % (self.data, self.g)


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

    def build(self):
        self.real_cost = 0
        return self

    def update(self):
        for agent in self.agents:
            agent.perceive()
            next_city = agent.act()

            current = self.current_state.data

            if (next_city is None or current == next_city or
                    not self.incidence[current][next_city]):
                # This agent doesn't know what to do.
                continue

            # Compute the total cost of the path.
            self.real_cost += self.d[current][next_city]
            # Transit to next city.
            self.current_state = CityState(next_city, g=self.real_cost)


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
    print(__doc__, flush=True)
    # It's pretty awkward print's flush and the logger's are not synchronized.
    # Use a sleep to make sure it's done.
    time.sleep(.01)

    env = SimplePathFinding(initial_state=CityState(0))

    env.agents += [
        RoutePlanner(environment=env,
                     # Uses Iterative-deepening search.
                     search=art.searches.fringe.IterativeDeepening,
                     # Depths searched are 2 and 3.
                     search_params={'iterations': range(2, 4)},
                     # Can move to any city.
                     actions=list(range(len(SimplePathFinding.incidence))))
    ]

    try:
        env.live(n_cycles=5)

        logger.info('\nSolution found! :-)'
                    if env.current_state.is_goal
                    else '\nSolution not found. :-(')

    except KeyboardInterrupt:
        logger.info('canceled')


if __name__ == '__main__':
    main()
