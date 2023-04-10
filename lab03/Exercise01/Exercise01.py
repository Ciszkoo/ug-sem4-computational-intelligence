import math
import pygad
import numpy


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def fitness_func(solution, solution_idx):
    fitness = endurance(*solution)
    return fitness


class Exercise01:
    ga_instance = pygad.GA(
        gene_space={"low": numpy.float32(0), "high": numpy.float32(1)},
        num_generations=30,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=6,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=17
    )

    def run(self):
        self.ga_instance.run()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness))
