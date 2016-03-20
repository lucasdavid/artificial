from artificial.base import Environment
from artificial.base import solvers


class RomaniaEnvironment(Environment):
    def update(self):
        for agent in self.agents:
            agent.perceive()
            agent.act()


    def predict(self, agent, state):
        state
        agent.actions

# Arad Oradea Zerind Timisoara Lugoj Mehadia Dobeta
# Sibiu Fagaras Rimnicu-Vilcea Crai
cities = ['Arad', 'Oradea', 'Zerind', 'Timisoara', 'Lugoj',
          'Mehadia', 'Dobeta', 'Sibiu', 'Fagaras', 'Rimnicu Vilcea'
          'Craiova', 'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni',
          'Neamt', 'Iasi', 'Vaslui', 'Hirsova', 'Eforie']
g = {
    nodes: len(cities),
    labels: cities,
    edges: {
        0: {}
    },
}


def main():
    print('Romania route fiding example.')

    env = RomaniaEnvironment(
        solvers.ProblemSolverAgent(

        ),
        current_state = State(
            data=[]
        )
    )


if __name__ == '__main__':
    main()

