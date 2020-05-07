from abc import ABC, abstractclassmethod
from math import pi, sin
from random import gauss


class BaseClass(ABC):
    @property
    def max_itarations(self):
        return 1

    @abstractclassmethod
    def run_algorithm(self):
        pass

    @abstractclassmethod
    def rate(self):
        pass

    @abstractclassmethod
    def disturb(self):
        pass

    def _gaussian_variation(self, x_value):
        return x_value + gauss(0, 0.1)

    def g_function(self, x_value):
        exponential_element = (x_value - 0.1)/0.9
        exponential_element = -2 * exponential_element
        exponential_element = exponential_element ** 2

        first_element = 2 ** exponential_element

        sin_input_value = 5 * pi * x_value
        sin_value = sin(sin_input_value)

        second_element = sin_value ** 6

        result = first_element * second_element

        return result
