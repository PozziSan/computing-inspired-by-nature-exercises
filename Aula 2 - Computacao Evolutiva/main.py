from hill_climb import HillClimb
from interactive_hill_climb import InteractiveHillClimb
from probabilistc_hill_climb import ProbabilisticHillClimb
from simulated_annealing import SimulatedAnnealing

hill_climb = HillClimb()
interactive_hill_climb = InteractiveHillClimb()
probabilistic_hill_climb = ProbabilisticHillClimb()
simulated_annealing = SimulatedAnnealing()

if __name__ == '__main__':
    # print(hill_climb.run_algorithm())
    # interactive_hill_climb.run_algorithm()
    # probabilistic_hill_climb.run_algorithm()
    simulated_annealing.run_algorithm()
