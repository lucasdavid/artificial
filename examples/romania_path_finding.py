from artificial import agents, searches
from artificial.base import Environment, State
from artificial.base.helpers import Graph


class Romania:
    g = Graph(
        nodes=['Arad', 'Oradea', 'Zerind', 'Timisoara', 'Lugoj', 'Mehadia',
                'Dobeta', 'Sibiu', 'Fagaras', 'Rimnicu Vilcea', 'Craiova',
                'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 'Neamt', 'Iasi',
                'Vaslui', 'Hirsova', 'Eforie'],
        edges={
            0: {2: 75, 3: 118, 7: 140},
            1: {2: 71, 7: 151},
            2: {0: 75, 1: 71},
            3: {0: 118, 4: 111},
            4: {3: 111, 5: 75},
            5: {4: 75},
            7: {0: 140, 1: 151},
        })

    source_city_id = 0
    target_city_id = 11


class CityState(State):
    @property
    def is_goal(self):
        return self.data == Romania.target_city_id

    @property
    def h(self):
        return (Romania.g.edges[self.data][Romania.target_city_id]
                if Romania.target_city_id in Romania.g.edges[self.data]
                else 0)

    def __str__(self):
        return 'city: %d, g: %d' % (self.data, self.g)


class RomaniaPathFinding(Environment):
    real_cost = 0

    def update(self):
        for agent in self.agents:
            agent.perceive()
            next_city = agent.act()

            current = self.current_state.data

            if next_city is None or \
               current == next_city or \
               next_city not in Romania.g.edges[current]:
                # Invalid transition or non-existent road.
                continue

            self.current_state = CityState(next_city)
            self.real_cost += Romania.g.edges[current][next_city]


class RoutePlanner(agents.UtilityBasedAgent):
    def predict(self, state):
        current = state.data
        neighbors = [city for city in self.actions
                     if city in Romania.g.edges[current]]

        return [CityState(neighbor, parent=state, action=neighbor,
                          g=state.g + Romania.g.edges[current][neighbor])
                for neighbor in neighbors]


def main():
    print('===========================')
    print('Romania Path Finding Example')
    print('===========================\n')

    env = RomaniaPathFinding(
        initial_state=CityState(Romania.source_city_id))

    env.agents += [
        RoutePlanner(environment=env,
                     search=searches.GreedyBestFirstSearch,
                     actions=list(range(Romania.g.n_nodes)),
                     verbose=True)
    ]

    i, max_iterations = 0, 100

    print('Initial state: {%s}\n' % str(env.current_state))

    try:
        while i < max_iterations and not env.finished():
            i += 1
            print('Iteration %i' % i)
            env.update()

            print('Current state: {%s}\n' % str(env.current_state))

        print('Solution found! (cost: %.1f) :-)' % env.real_cost
              if env.current_state.is_goal
              else 'Solution not found. :-(')

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
