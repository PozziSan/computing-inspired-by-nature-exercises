#procedimento [𝐱] = stochastic-hill-climbing(max_it, g)
# inicializar 𝐱
# avaliar(𝐱)
# t ← 1
# enquanto t < max_it & avaliar(𝐱) != g faça
# 𝐱’ ← perturbar(𝐱)
# avaliar(𝐱’)
# se aleatorio[0,1) < (1/(1+exp[(avaliar(𝐱)-avaliar(𝐱’))/𝑇]))
# então 𝐱 ← 𝐱’
# fim-se
# t ← t + 1
# fim-enquanto
# fim-procedimento

from hill_climb import HillClimb
from math import exp
from random import random


class ProbabilisticHillClimb(HillClimb):
    def run_algorithm(self):
        print(self._probabilistic_hill_climbing(self.max_itarations, 1))

    def _probabilistic_hill_climbing(self, max_it, best_value):
        decaying_rate_element = 15
        x = 0
        for _ in range(max_it):
            x_rate = self.rate(x)

            new_x = self.disturb(x)
            new_x_rate = self.rate(new_x)

            random_value = random()

            exponential_function_result = (
                1 / (1 + exp(x_rate - new_x_rate)) / decaying_rate_element)

            if random_value < exponential_function_result:
                x = new_x
                x_rate = new_x_rate

            if x_rate == best_value:
                break

        return x
