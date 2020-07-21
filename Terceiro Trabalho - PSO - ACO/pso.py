import random
import matplotlib.pyplot as plt
import numpy as np


# funcao de aptidao
def fitness_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# classe das particulas
class Particle:
    def __init__(self, dimension):
        self.dimension = dimension
        self.position = []
        self.velocity = []
        self.best_position = []
        self.fitness = None
        self.best_fitness = -1 #inicia em -1 pra entrar na condicional mais abaixo

        for _ in range(dimension): # nesse problema a dimensão é 2, então nem precisaria disso aqui, mas fica pelo exemplo de como seria um mais genérico..
            self.position.append(random.uniform(-5, 5)) #inicia entre -5 e 5
            self.velocity.append(random.uniform(0, 3)) # inicia entre 0 e 3

    def evaluate(self):
        x = self.position[0]
        y = self.position[1]

        self.fitness = fitness_function(x, y)

        if self.fitness < self.best_fitness or self.best_fitness == -1:  #seta a melhor classificacao desse individuo  (se for a primeira execução, seta ele como melhor, graças ao -1 iniciado no construtor
            self.best_position = self.position
            self.best_fitness = self.fitness

    def update_velocity(self, best_group_position):
        w = 0.5  # constante de inércia
        c1 = 1  # constante de cognico
        c2 = 2  # constante social

        for i in range(0, self.dimension):
            r1 = random.random() #numero aleatorio 1
            r2 = random.random() #numero aleatorio 2

            cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (best_group_position[i] - self.position[i])

            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self):
        for i in range(0, self.dimension):
            self.position[i] = self.position[i] + self.velocity[i]


# classe do optimizador de particulas
class PSO:
    def __init__(self, particle_dimension, bounds, number_of_particles, max_iterations):
        #atributos padrao
        self.particle_dimension = particle_dimension
        self.bounds = bounds
        self.number_of_partticles = number_of_particles
        self.max_iterations = max_iterations

        #atributos do grupo de particulas
        self.best_group_position = []
        self.best_group_fitness = -1
        self.swarm = []

        #atributos para estatísticas
        self.particle_positions = []
        self.particle_fitness = []

        self.particle_mean_positions_per_generation = []
        self.particle_min_positions_per_generation = []

        self.particle_mean_fitness_per_generation = []
        self.particle_min_fitness_per_generation = []

        self.best_position_list_per_generation = []
        self.best_fitness_list_per_generation = []

    #inicializa o grupo
    def initialize_swarm(self):
        for _ in range(self.number_of_partticles):
            self.swarm.append(Particle(self.particle_dimension))

    #executa a optimizacao
    def optimize(self):
        for _ in range(self.max_iterations): #inicia o loop pelo numero maximo de iteracoes (geracoes)
            for particle in self.swarm:
                particle.evaluate() #avalia a particula

                if particle.fitness < self.best_group_fitness or self.best_group_fitness == -1: #se ela for melhor que a melhor do grupo, efetua a atualizacao
                    self.best_group_position = list(particle.position)
                    self.best_group_fitness = float(particle.fitness)

            for particle in self.swarm: #apos avaliadas todas as particulas e definidas a melhor da geracao, atualiza a velocidade e posicao das outras particulas
                self.particle_fitness.append(particle.fitness)
                self.particle_positions.append(particle.position)

                particle.update_velocity(self.best_group_position)
                particle.update_position()

            # adiciona os valores da geração as estatísticas
            self.best_position_list_per_generation.append(self.best_group_position)
            self.best_fitness_list_per_generation.append(self.best_group_fitness)

            self.particle_mean_positions_per_generation.append(np.mean(self.particle_positions))
            self.particle_min_positions_per_generation.append(np.min(self.particle_positions))

            self.particle_mean_fitness_per_generation.append(np.mean(self.particle_fitness))
            self.particle_min_fitness_per_generation.append(min(self.particle_fitness))


# funcao pra plotar o grafico (plotar é um termo estranho..)
def plot(min_values, mean_values):
    plt.plot(mean_values, color='blue')
    plt.plot(min_values, color='red')

    plt.legend(['media', 'mínimos'], loc='upper right')
    plt.ylabel('Aptidão')
    plt.xlabel('Gerações')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    bounds = [10, 10]
    pso = PSO(2, bounds, 10, 500)

    pso.initialize_swarm()
    pso.optimize()

    print(pso.best_group_position)
    print(pso.best_group_fitness)
    print(pso.particle_min_fitness_per_generation)
    print(pso.particle_mean_fitness_per_generation)

    plot(pso.particle_min_fitness_per_generation, pso.particle_mean_fitness_per_generation)
