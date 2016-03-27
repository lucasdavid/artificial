from artificial import base, agents
from artificial.searches import fringe as searches

class CityState(base.State):
    """City State.

    CityState doesn't need to override State's __eq__ and __hash__ methods,
    as two city states are the same if they've achieved the same city and,
    hence, have the same `data` attribute, which is precisely what
    State.__eq__ checks for.
    """

    @property
    def is_goal(self):
        return self.data == 3

    def __str__(self):
        return 'city: %d, g: %d' % (self.data, self.g)


class SimplePathFinding(base.Environment):
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

            if next_city is None or \
               current == next_city or \
               not self.incidence[current][next_city]:
                # This agent doesn't know what to do.
                continue

            self.current_state = CityState(next_city)
            self.real_cost += self.d[current][next_city]


class RoutePlanner(agents.UtilityBasedAgent):
    def predict(self, state):
        current = state.data
        d, roads = self.environment.d, self.environment.incidence
        neighbors = [city for city in self.actions
                     if roads[current][city]]

        return [CityState(data=neighbor, parent=state, action=neighbor,
                          g=state.g + d[current][neighbor])
                for neighbor in neighbors]


def main():
    print('===========================')
    print('Simple Path Finding Example')
    print('===========================\n')

    env = SimplePathFinding(initial_state=CityState(0))

    env.agents += [
        RoutePlanner(environment=env,
                     # Uses Iterative-deepening search.
                     search=searches.IterativeDeepening,
                     # Depths searched are 2 and 3.
                     search_params={'iterations': range(2, 4)},
                     # Can move to any city.
                     actions=list(enumerate(SimplePathFinding.g)),
                     # Talks all the way.
                     verbose=True)
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
