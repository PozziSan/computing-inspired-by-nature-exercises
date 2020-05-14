# procedimento [𝐱] = hill-climbing(max_it,g)
# inicializar 𝐱
# avaliar(𝐱)
# t ← 1
# enquanto t < max_it & avaliar(𝒙) != g faça
# 𝐱’ ← perturbar(𝐱)
# avaliar(𝐱’)
# se avaliar(𝐱’) é melhor que avaliar(x)
# então 𝐱 ← 𝐱’
# fim-se
# t ← t + 1
# fim-enquanto
# fim-procedimento

from base import BaseClass


class HillClimb(BaseClass):
    @property
    def algorithm_name(self):
        return 'Hill Climb'

    @property
    def max_iterations(self):
        return 1000

    def run_algorithm(self):
        print(self._hill_climbing(self.max_iterations, 1))

    def rate(self, x_value):
        return self.g_function(x_value)

    def disturb(self, x_value):
        return self._gaussian_variation(x_value)

    def _hill_climbing(self, max_it, best_value):
        x = 0

        for _ in range(max_it):
            x_rate = self.rate(x)

            new_x_value = self.disturb(x)
            new_x_value_rate = self.rate(new_x_value)

            if new_x_value_rate > x_rate:
                x = new_x_value
                x_rate = new_x_value_rate

            self.append_element(x_rate)

        return x
