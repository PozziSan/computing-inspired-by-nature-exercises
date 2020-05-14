from abc import ABC, abstractclassmethod, abstractmethod
from matplotlib import pyplot
from math import pi, sin
from random import gauss
from numpy import std


class BaseClass(ABC):
    def __init__(self):
        self.value_per_iteration = []

    def append_element(self, element):
        self.value_per_iteration = self.value_per_iteration + [element]

    @property
    def max_iterations(self):
        return 1

    @property
    @abstractmethod
    def algorithm_name(self):
        pass

    @abstractmethod
    def run_algorithm(self):
        pass

    @abstractmethod
    def rate(self):
        pass

    @abstractmethod
    def disturb(self):
        pass

    def _gaussian_variation(self, x_value):
        return x_value + gauss(0, 0.1)

    def g_function(self, x_value):
        exponential_element = (x_value - 0.1)/0.9
        exponential_element = exponential_element ** 2
        exponential_element = -2 * exponential_element

        first_element = 2 ** exponential_element

        sin_input_value = 5 * pi * x_value
        sin_value = sin(sin_input_value)

        second_element = sin_value ** 6

        result = first_element * second_element

        return result

    def plot_graphic(self):
        pyplot.plot(self.value_per_iteration, color='red')

        pyplot.ylabel('Resultado da função')
        pyplot.xlabel('Iteração')
        pyplot.grid(True)
        pyplot.title(self.algorithm_name)

        pyplot.show()

    def standard_deviation(self):
        return std(self.value_per_iteration)