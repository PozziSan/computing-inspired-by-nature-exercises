#procedimento [𝐱] = simulated_annealing(g)
# inicializar T
# inicializar 𝐱
# avaliar(𝐱)
# t ← 1
# enquanto não atender critério de parada faça
# 𝐱’ ← perturbar(𝐱)
# avaliar(𝐱’)
# se avaliar(𝐱’) é melhor que avaliar(𝐱),
# então 𝐱 ← 𝐱’
# senão se aleatorio[0,1) < exp[(avaliar(𝐱’)-avaliar(𝐱))/𝑇],
# então 𝐱 ← 𝐱’
# fim-se
# 𝑇 ← g(𝑇,t)
# t ← t + 1
# fim-enquanto
# fim-procedimento


from base import BaseClass
from math import exp
from random import random


class SimulatedAnnealing(BaseClass):
    @property
    def max_iterations(self):
        return 100

    def run_algorithm(self):
        print(self._simmulated_annealing())

    def rate(self, x_value):
        return self.g_function(x_value)

    def disturb(self, x_value):
        return self._gaussian_variation(x_value)

    def _init_values(self):
        x, system_temperature = 0, 1

        return x, system_temperature

    def _disturb_system(self, x, system_temperature):
        system_temperature = self.g_function(system_temperature)
        x_rate = self.rate(x)
        best_value = 1

        for _ in range(self.max_iterations):
            new_x = self.disturb(x)
            new_x_rate = self.rate(new_x)

            random_element = random()
            exponential_function_result = exp(
                (new_x_rate - x_rate) / system_temperature)

            if abs(best_value - new_x_rate) < abs(best_value - x_rate):
                x, x_rate = new_x, new_x_rate
            elif random_element < exponential_function_result:
                x, x_rate = new_x, new_x_rate

        return x, system_temperature

    def _simmulated_annealing(self):
        x, system_temperature = self._init_values()

        min_system_temperature = 1

        system_temperature = 0.9 * system_temperature
        print(system_temperature)

        x, system_temperature = self._disturb_system(x, system_temperature)

        print(f'x: {x}, system_temperature: {system_temperature}')

        if system_temperature >= min_system_temperature:
            x, system_temperature = self._disturb_system(x, system_temperature)
        else:
            x, system_temperature = self._init_values()

        return x, system_temperature
