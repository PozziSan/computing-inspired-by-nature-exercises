from hill_climb import HillClimb
from interactive_hill_climb import InteractiveHillClimb
from probabilistc_hill_climb import ProbabilisticHillClimb

hill_climb = HillClimb()
interactive_hill_climb = InteractiveHillClimb()
probabilistic_hill_climb = ProbabilisticHillClimb()

if __name__ == '__main__':
    # print(hill_climb.run_algorithm())
    # interactive_hill_climb.run_algorithm()
    probabilistic_hill_climb.run_algorithm()
