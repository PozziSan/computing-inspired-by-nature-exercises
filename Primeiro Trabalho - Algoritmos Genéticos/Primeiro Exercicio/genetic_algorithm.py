from numpy import sum, mean, std
from random import random, randint
from matplotlib import pyplot


bitstring = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
element_len = len(bitstring)
population_len = 60

crossover_rate = 0.85
mutation_rate = 0.0001

number_of_generations = 100
number_of_elements_selected_by_generation = 30


def init_population():
    population = []

    for _ in range(population_len):
        element = []
        for _ in range(element_len):
            element.append(randint(0, 1))

        population.append(element)

    return population


def score_elements(population):
    """

    @type population: list
    @return element_score: list
    """
    elements_score = []

    for element in population:
        gene_capabilities = 0
        for index, gene, in enumerate(element):
            if gene == bitstring[index]:
                gene_capabilities += 1

        elements_score.append(gene_capabilities)

    return elements_score


def make_wheel(elements_score):
    wheel = []
    total = sum(elements_score)
    top = 0

    for element_index, element_score in enumerate(elements_score):
        element_percent = element_score / total

        wheel.append((top, top + element_percent, element_index))
        top += element_percent

    return wheel


def binary_search(wheel, num):
    mid = len(wheel) // 2
    low, high, answer_index = wheel[mid]

    if low <= num <= high:
        return answer_index
    elif high < num:
        return binary_search(wheel[mid+1:], num)
    else:
        return binary_search(wheel[:mid], num)


def select_parents(wheel, number_of_elements_to_select):
    step_size = 1.0 / number_of_elements_to_select
    parents_index = []

    wheel_position = random()
    parents_index.append(binary_search(wheel, wheel_position))

    while len(parents_index) < number_of_elements_to_select:
        wheel_position += step_size

        if wheel_position > 1:
            wheel_position %= 1

        parents_index.append(binary_search(wheel, wheel_position))

    return parents_index


def crossover(matching_parents):
    offspring = []
    first_parent, second_parent = None, None

    for parent_index, parent in enumerate(matching_parents):
        if parent_index % 2 == 0:
            first_parent = parent
        else:
            second_parent = parent

            crossover_chance = random()

            if crossover_chance < crossover_rate:
                first_son, second_son = crossover_by_two_parents(first_parent, second_parent)

                offspring.append(first_son)
                offspring.append(second_son)

    return offspring


def crossover_by_two_parents(first_parent, second_parent):
    crossover_point = randint(0, element_len)

    first_son = first_parent[0:crossover_point] + second_parent[crossover_point:]
    second_son = second_parent[0:crossover_point] + first_parent[crossover_point:]

    return first_son, second_son


def mutate(population):
    for element_index, element in enumerate(population):
        for gene_index, gene in enumerate(element):
            random_value = random()

            if random_value < mutation_rate:
                population[element_index][gene_index] = 0 if population[element_index][gene_index] == 1 else 1

    return population


def reproduce_population(population, parents, offspring):
    if offspring:
        for parent in parents:
            if parent in population:
                population.remove(parent)

        for element in offspring:
            population.append(element)

    return population


def score_generation(population):
    elements_score = score_elements(population)

    score_sum = sum(elements_score)
    score = score_sum / len(population)

    return score


if __name__ == '__main__':
    population = init_population()
    best_min = []
    best_mean = []
    best_max = []
    best_max_executions = []
    best_mean_executions = []
    best_min_executions = []
    elements_score_executions = []

    for _ in range(1000):
        generations_score = []
        for _ in range(number_of_generations):
            element_score = score_elements(population)

            wheel = make_wheel(element_score)
            parents_index = select_parents(wheel, number_of_elements_selected_by_generation)
            parents = [population[element] for element in parents_index]

            offspring = crossover(parents)

            population = reproduce_population(population, parents, offspring)

            population = mutate(population)

            generations_score.append(score_generation(population))

            best_min.append(min(element_score))
            best_mean.append(mean(element_score))
            best_max.append(max(element_score))


        elements_score_executions.append(generations_score)

        best_max_executions.append(mean(best_max))
        best_mean_executions.append(mean(best_mean))
        best_min_executions.append(mean(best_min))



    pyplot.plot(best_max_executions, color='blue', label='Média dos Máximos')
    pyplot.plot(best_mean_executions, color='green', label='Média dos Médios')
    pyplot.plot(best_min_executions, color='red', label='Média dos Mínimos')

    pyplot.ylabel('Aptidão')
    pyplot.xlabel('Gerações')
    pyplot.title('Estatísticas')
    pyplot.grid(True)
    pyplot.legend(loc='lower right')

    pyplot.show()

    print(std(elements_score_executions))
