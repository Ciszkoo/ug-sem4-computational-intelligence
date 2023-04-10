import numpy as np
from Exercise01 import Exercise01
from Exercise02 import Exercise02

if __name__ == '__main__':
    print("Exercise 1:")
    Exercise01.Exercise01().run()

    print("Exercise 2:")
    Exercise02.Exercise02().run()

    timings = []

    while len(timings) != 10:
        fit, solution, timing = Exercise02.Exercise02().run()
        if fit == 0:
            timings.append(timing)
            print(f"Fitness value of the best solution: {fit}")
            print(f"Parameters of the best solution: {solution}")
            print(f"Running time of the algorithm: {timing}\n")

    mean = np.mean(timings)
    print(f"Average time of finding a path through the maze: {mean}")
