import math
import operator
import random

import matplotlib.pyplot as plt


class Graph:
    def __init__(self, cost_matrix, rank):
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank ** 2) for _ in range(rank)] for _ in range(rank)]


class ACO:
    def __init__(self, ant_count, generations, alpha, beta, pheromone_residual_coefficient, pheromone_intensity):
        self.pheromone_intensity = pheromone_intensity
        self.pheromone_residual_coefficient = pheromone_residual_coefficient
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.best_cost_per_generations = []

    def update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.pheromone_residual_coefficient

                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph):
        best_cost = float('inf')  # numero infinito
        best_solution = []

        for _ in range(self.generations):
            ants = [Ant(self, graph) for _ in range(self.ant_count)]  # instancia formigas

            for ant in ants:
                for _ in range(graph.rank - 1):
                    ant._select_next()

                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]

                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.tabu

                # atualiza delta feromonio
                ant._update_pheromone_delta()

            self.update_pheromone(graph, ants)
            self.best_cost_per_generations.append(best_cost)

        return best_solution, best_cost


class Ant:
    def __init__(self, aco, graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # lista de tabu
        self.pheromone_delta = []  # diferença local de feromonio
        self.allowed = [i for i in range(graph.rank)]  # nos permitidos para a selecao
        self.eta = [[0 if i == j else (1 / graph.matrix[i][j]) for j in range(graph.rank)] for i in
                    range(graph.rank)]  # informacao heuristica

        start = random.randint(0, graph.rank - 1)  # comeca de qualquer nó
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0

        for node in self.allowed:
            denominator += self.graph.pheromone[self.current][node] ** self.colony.alpha * self.eta[self.current][
                node] ** self.colony.beta

        probabilities = [0 for _ in range(self.graph.rank)]  # probabilidade de mover para o proximo no

        for node in range(self.graph.rank):
            try:
                self.allowed.index(node)  # test if allowed list contains i
                probabilities[node] = self.graph.pheromone[self.current][node] ** self.colony.alpha * \
                                      self.eta[self.current][node] ** self.colony.beta / denominator
            except ValueError as e:
                pass

        # seleciona o proximo no e acordo com probabilidade
        selected = 0
        rand = random.random()

        for index, probability in enumerate(probabilities):
            rand -= probability

            if rand <= 0:
                selected = index
                break

        # remove no atual
        self.allowed.remove(selected)
        self.tabu.append(selected)
        # adiciona o atual ao custo total
        self.total_cost += self.graph.matrix[self.current][selected]
        # atualiza o no atual
        self.current = selected

    def _update_pheromone_delta(self):
        # cria matriz de delta feromonio
        self.pheromone_delta = [[0 for _ in range(self.graph.rank)] for _ in range(self.graph.rank)]

        for index in range(1, len(self.tabu)):
            i = self.tabu[index - 1]
            j = self.tabu[index]

            self.pheromone_delta[i][j] = self.colony.pheromone_intensity


# funcao pra plotar o grafico  (plotar é um termo mt estranho em portugues...)
def plot(points, path):
    x, y = [], []

    points_len = len(points)
    path_len = len(path)

    for point in points:
        x.append(point[0])
        y.append(point[1])

    max_y_list = [max(y) for _ in range(points_len)]
    # cria os valores de Y, no caso, uso o valor maximo de Y  - o valor original de y
    y = list(map(operator.sub, max_y_list, y))  # preciso usar a função list() para converter o gerador em lista

    plt.plot(x, y, 'co')

    for index in range(1, path_len):
        i = path[index - 1]
        j = path[index]

        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)

    plt.show()


def plot_cost_graph(best_cost_per_generations):
    plt.plot(best_cost_per_generations, color='red')

    plt.ylabel('Custo')
    plt.xlabel('Gerações')

    plt.grid(True)

    plt.show()

def distance(city1, city2):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def parse_file(file_path, ignored_lines):
    cities, points = [], []

    with open(file_path, 'r') as tsp_file:
        for index, line in enumerate(tsp_file.readlines()):
            if index in ignored_lines:
                pass
            else:
                city = line.split(' ')
                city_dict = {
                    'index': int(city[0]),
                    'x': float(city[1]),
                    'y': float(city[2])
                }
                point_tuple = (float(city[1]), float(city[2]))

                cities.append(city_dict)
                points.append(point_tuple)

    return cities, points


def generate_graph_values(cities):
    cost_matrix = []
    rank = len(cities)

    for i in range(rank):
        row = []
        for j in range(rank):
            row_distance = distance(cities[i], cities[j])
            row.append(row_distance)

        cost_matrix.append(row)

    return cost_matrix, rank


if __name__ == '__main__':
    file_path = 'berlin52.tsp'
    ignored_lines = [0, 1, 2, 3, 4, 5, 58, 59, 60]

    cities, points = parse_file(file_path, ignored_lines)
    cost_matrix, rank = generate_graph_values(cities)

    aco = ACO(20, 10, 1.0, 10.0, 0.5, 10)
    graph = Graph(cost_matrix, rank)

    path, cost = aco.solve(graph)
    best_cost_per_generations = aco.best_cost_per_generations

    print(best_cost_per_generations)

    print('cost: {}, path: {}'.format(cost, path))
    plot(points, path)
    plot_cost_graph(best_cost_per_generations)
