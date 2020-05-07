# procedimento [] = IHC(n_start, max_it, g)
# inicializar melhor
# t1 ← 1
# enquanto t1 < n_start & avaliar(melhor) != g faça,
# iniclizar 𝐱
# avaliar(𝐱)
# 𝐱 ← hill-climbing(max_it, g) // Hill Climbing Original
# t1 ← t1 + 1
# se avaliar(𝐱) é melhor que avaliar(melhor),
# então melhor ← 𝐱
# fim-se
# fim-enquanto
# fim-procedimento

from hill_climb import HillClimb


class InteractiveHillClimb(HillClimb):
    def run_algorithm(self):
        print(self._interactive_hill_climbing(3, self.max_itarations, 1))

    def _interactive_hill_climbing(self, n_start, max_it, best_value):
        better_value = self.g_function(1)
        better_value_rate = self.rate(better_value)

        for _ in range(n_start):
            x_value = self._hill_climbing(max_it, best_value)
            x_value_rate = self.rate(x_value)

            if abs(best_value - x_value_rate) < abs(best_value - better_value_rate):
                better_value = x_value
                better_value_rate = x_value_rate

            if better_value_rate == best_value:
                break

        return x_value
