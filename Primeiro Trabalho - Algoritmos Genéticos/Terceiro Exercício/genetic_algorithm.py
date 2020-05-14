import math
import random

from matplotlib import pyplot
from numpy import sum, mean

population_len = 100

crossover_rate = 0.7
mutation_rate = 0.01

number_of_generations = 1000
number_of_elements_selected_by_generation = 60


def generate_random_element():
    return {
        'x': round(random.uniform(-5, 5), 7),
        'y': round(random.uniform(-5, 5), 7)
    }


def rating_function(_population):
    list_of_results = []

    for element in _population:
        x, y = element.values()

        result = round((1 - x) ** 2 + (100 * (y - (x ** 2)) ** 2), 7)
        list_of_results.append(result)

    return list_of_results


def init_population():
    _population = [generate_random_element() for _ in range(population_len)]

    return _population


def fitness_result(element_result):
    return round(1 / (element_result + 1), 7)


def score_elements(population_results):
    elements_score = [fitness_result(result) for result in population_results]

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
        return binary_search(wheel[mid + 1:], num)
    else:
        return binary_search(wheel[:mid], num)


def select_parents(wheel, number_of_elements_to_select):
    step_size = 1.0 / number_of_elements_to_select
    parents_index = []

    wheel_position = random.random()
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

            crossover_chance = random.random()

            if crossover_chance < crossover_rate:
                first_son, second_son = crossover_by_two_parents(first_parent, second_parent)

                offspring.append(first_son)
                offspring.append(second_son)

    return offspring


def crossover_by_two_parents(first_parent, second_parent):
    x_or_y = random.choice(['x', 'y'])

    if x_or_y is 'x':
        first_son = {
            'x': first_parent['x'] + second_parent['x'],
            'y': first_parent['y']
        }

        second_son = {
            'x': first_parent['x'],
            'y': first_parent['y'] + second_parent['y']
        }

        first_son['x'] = normalize_value_within_problem_scope(first_son['x'])
        second_son['y'] = normalize_value_within_problem_scope(second_son['y'])
    else:
        first_son = {
            'x': first_parent['x'],
            'y': first_parent['y'] + second_parent['y']
        }

        second_son = {
            'x': first_parent['x'] + second_parent['x'],
            'y': first_parent['y']
        }

        first_son['y'] = normalize_value_within_problem_scope(first_son['y'])
        second_son['x'] = normalize_value_within_problem_scope(second_son['x'])

    return first_son, second_son


def normalize_value_within_problem_scope(value):
    if value > 5:
        value = 5
    elif value < -5:
        value = -5

    return value


def mutate_integer_value(element, gene_key):
    truncated_value = math.trunc(element[gene_key])

    if truncated_value == -5:
        truncated_value += 1
    elif truncated_value == 5:
        truncated_value -= 1
    else:
        truncated_value += random.choice([-1, 1])

    module = round(element[gene_key] % 1, 7)
    new_value = truncated_value + module

    new_value = normalize_value_within_problem_scope(new_value)

    return new_value


def mutate_decimal_value(element, gene_key):
    integer_value = truncated_value = math.trunc(element[gene_key])
    module = round(element[gene_key] % 1, 7)
    random_decimal_digit = random.randint(0, 9)

    new_value = round(module - random_decimal_digit)
    new_value += integer_value

    new_value = normalize_value_within_problem_scope(new_value)

    return new_value


def mutate(population):
    for element_index, element in enumerate(population):
        for gene_key in element.keys():
            random_value = random.random()

            if random_value < mutation_rate:
                integer_or_decimal = random.choice(['integer', 'decimal'])

                if integer_or_decimal is 'integer':
                    new_value = mutate_integer_value(element, gene_key)
                else:
                    new_value = mutate_decimal_value(element, gene_key)

                population[element_index][gene_key] = new_value

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
    population_results = rating_function(population)
    elements_score = score_elements(population_results)

    score_sum = sum(elements_score)
    score = score_sum / len(population)

    return score


if __name__ == '__main__':
    population = init_population()
    generations_score = []
    best_min = []
    best_mean = []
    best_max = []
    best_min_results = []
    best_mean_results = []
    best_max_results = []

    for _ in range(number_of_generations):
        population_results = rating_function(population)
        elements_score = score_elements(population_results)

        wheel = make_wheel(elements_score)
        selected_parents_index = select_parents(wheel, number_of_elements_selected_by_generation)
        selected_parents = [population[index] for index in selected_parents_index]

        offspring = crossover(selected_parents)
        population = reproduce_population(population, selected_parents, offspring)
        population = mutate(population)

        best_min_results.append(min(population_results))
        best_mean_results.append(mean(population_results))
        best_max_results.append(max(population_results))

        best_min.append(min(elements_score))
        best_mean.append(mean(elements_score))
        best_max.append(max(elements_score))

    pyplot.plot(best_max, color='blue')
    pyplot.plot(best_mean, color='green')
    pyplot.plot(best_min, color='red')

    pyplot.ylabel('Aptidão')
    pyplot.xlabel('Gerações')
    pyplot.title('Elements Score')
    pyplot.grid(True)

    pyplot.show()

    pyplot.plot(best_max_results, color='blue')
    pyplot.plot(best_mean_results, color='green')
    pyplot.plot(best_min_results, color='red')

    pyplot.ylabel('Aptidão')
    pyplot.xlabel('Gerações')
    pyplot.title('Results')
    pyplot.grid(True)

    pyplot.show()
