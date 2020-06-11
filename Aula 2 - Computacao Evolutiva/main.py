# -*- coding: utf-8 -*-

from hill_climb import HillClimb
from simulated_annealing import SimulatedAnnealing
# from interactive_hill_climb import InteractiveHillClimb

hill_climb = HillClimb()
# interactive_hill_climb = InteractiveHillClimb()
# probabilistic_hill_climb = ProbabilisticHillClimb()
simulated_annealing = SimulatedAnnealing()

if __name__ == '__main__':
    # print(hill_climb.run_algorithm())
    # hill_climb.plot_graphic()
    # interactive_hill_climb.run_algorithm()
    # interactive_hill_climb.plot_graphic()
    # probabilistic_hill_climb.run_algorithm()
    simulated_annealing.run_algorithm()
    simulated_annealing.plot_graphic()
