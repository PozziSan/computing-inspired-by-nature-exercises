from random import random, randint
from bitstring import BitArray
from numpy import sum, mean, nan, std
from math import sin, pi, isnan
from matplotlib import pyplot

population_len = 60
element_len = 32

number_of_elements_selected_by_generation = 30
number_of_generations = 100

crossover_rate = 0.85
mutation_rate = 0


def init_population():
    population = []

    for _ in range(population_len):
        random_float = random()
        random_bin = BitArray(float=random_float, length=32)
        element = [int(gene) for gene in random_bin.bin]

        population.append(element)

    return population


def convert_population_to_float(population):
    float_population = []

    for element_index, element in enumerate(population):
        element_str = ''.join(str(bit) for bit in element)
        element_bitstring = BitArray(bin=element_str)
        element_float = element_bitstring.float

        normalized_element = normalize_element_within_scope(element_float)

        population[element_index] = [int(gene) for gene in normalized_element.bin]

        float_population.append(normalized_element.float)

    return population, float_population


def normalize_element_within_scope(element_float):
    if element_float > 1:
        fixed_element = BitArray(int=1, length=32)
    elif element_float < 0:
        fixed_element = BitArray(int=0, length=32)
    elif isnan(element_float):
        fixed_element = random()
        fixed_element = BitArray(float=fixed_element, length=32)
    else:
        fixed_element = BitArray(float=element_float, length=32)

    return fixed_element

def function_result(x_float_value):
    exponential_element = (x_float_value - 0.1) / 0.9
    exponential_element = exponential_element ** 2
    exponential_element = -2 * exponential_element

    first_element = 2 ** exponential_element

    sin_input_value = 5 * pi * x_float_value
    sin_value = sin(sin_input_value)

    second_element = sin_value ** 6

    result = first_element * second_element

    return result


def population_result(population_float_values):
    return [function_result(element) for element in population_float_values]


def fitness_function(element_result, population_sum):
    return element_result / population_sum


def score_elements(population_result, population_result_sum):
    elements_score = [fitness_function(element_result, population_result_sum) for element_result in population_result]

    if nan in elements_score:
        print(population_result)

        raise Exception('o porra')

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

    first_son_str = ''.join(str(gene) for gene in first_son)
    second_son_str = ''.join(str(gene) for gene in second_son)
    first_son_bitstring = BitArray(bin=first_son_str)
    second_son_bitstring = BitArray(bin=second_son_str)

    normalized_first_son = normalize_element_within_scope(first_son_bitstring.float)
    normalized_second_son = normalize_element_within_scope(second_son_bitstring.float)

    first_son = [int(gene) for gene in normalized_first_son.bin]
    second_son = [int(gene) for gene in normalized_second_son.bin]

    return first_son, second_son


def mutate(population):
    for element_index, element in enumerate(population):
        new_element = []
        for gene_index, gene in enumerate(element):
            random_value = random()

            if random_value < mutation_rate:
                mutated_gene = 0 if gene == 1 else 1

                new_element.append(mutated_gene)
            else:
                new_element.append(gene)

        if new_element != element:
            mutated_element = ''.join(str(gene) for gene in new_element)
            mutated_element_bitstring = BitArray(bin=mutated_element)

            normalized_mutated_element = normalize_element_within_scope(mutated_element_bitstring.float)

            population[element_index] = [int(gene) for gene in normalized_mutated_element.bin]

    return population


def reproduce_population(population, parents, offspring):
    if offspring:
        for parent in parents:
            if parent in population:
                population.remove(parent)

        for element in offspring:
            population.append(element)

    return population


if __name__ == '__main__':
    # Gera população inicial aleatoriamente
    population = init_population()
    best_min = []
    best_mean = []
    best_max = []
    best_min_results = []
    best_mean_results = []
    best_max_results = []
    best_max_execution = []
    best_min_execution = []
    best_mean_execution = []
    best_max_results_execution = []
    best_min_results_execution = []
    best_mean_results_execution = []

    # Laço de Repetição para executar o algoritmo 100 vezes
    for _ in range(100):
        # Laço de repetição para executar N gerações
        for _ in range(number_of_generations):
            # Converte os valores da população para float
            population, population_float_values = convert_population_to_float(population)
            # Avalia os resultados dos individuos
            population_results = population_result(population_float_values)

            # realiza a soma dos resultados dos individuos
            population_sum = sum(population_results)
            # avalia os individuos
            elements_score = score_elements(population_results, population_sum)

            #cria a roleta de seleção
            wheel = make_wheel(elements_score)
            # encontra os indices de pais selecionados
            parents_index = select_parents(wheel, number_of_elements_selected_by_generation)
            # cria o array de pais
            parents = [population[element] for element in parents_index]

            #cria os filhos
            offspring = crossover(parents)
            #insere os filhos no lugar dos pais
            population = reproduce_population(population, parents, offspring)

            #aplica a mutação. (a checagem é feita dentro da função)
            population = mutate(population)

            # adiciona as estatisticas
            best_min_results.append(min(population_results))
            best_mean_results.append(mean(population_results))
            best_max_results.append(max(population_results))

            best_min.append(min(elements_score))
            best_mean.append(mean(elements_score))
            best_max.append(max(elements_score))

        best_max_execution.append(mean(best_max))
        best_mean_execution.append(mean(best_mean))
        best_min_execution.append(mean(best_min))

        best_max_results_execution.append(mean(best_max_results))
        best_mean_results_execution.append(mean(best_mean_results))
        best_min_results_execution.append(mean(best_min_results))

    pyplot.plot(best_max_execution, color='blue', label='Média de Máximos')
    pyplot.plot(best_mean_execution, color='green', label='Média de Médios')
    pyplot.plot(best_min_execution, color='red', label='Média de Mínimos')

    pyplot.ylabel('Aptidão')
    pyplot.xlabel('Gerações')
    pyplot.title('Aptidão dos Elementos')
    pyplot.grid(True)
    pyplot.legend(loc='lower right')

    pyplot.show()

    pyplot.plot(best_max_results_execution, color='blue', label='Média de Máximos')
    pyplot.plot(best_mean_results_execution, color='green', label='Média de Médios')
    pyplot.plot(best_min_results_execution, color='red', label='Média de Mínimos')

    pyplot.ylabel('Resultado da Função')
    pyplot.xlabel('Gerações')
    pyplot.title('Resultados da Função por Indivíduo')
    pyplot.legend(loc='lower right')
    pyplot.grid(True)

    pyplot.show()

    print(std(best_mean_execution))