import time
import pygad

maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def fitness_func(solution, solution_idx):
    exit_position = [10, 10]
    actual_position = [1, 1]

    for move in solution:
        if move == 0 and maze[actual_position[0]][actual_position[1] - 1] == 1:
            actual_position[1] = actual_position[1] - 1
        if move == 1 and maze[actual_position[0] - 1][actual_position[1]] == 1:
            actual_position[0] = actual_position[0] - 1
        if move == 2 and maze[actual_position[0]][actual_position[1] + 1] == 1:
            actual_position[1] = actual_position[1] + 1
        if move == 3 and maze[actual_position[0] + 1][actual_position[1]] == 1:
            actual_position[0] = actual_position[0] + 1
        if actual_position[0] == exit_position[0] and actual_position[1] == exit_position[1]:
            return 0

    fitness = -((exit_position[0] - actual_position[0]) + (exit_position[1] - actual_position[1]))
    return fitness


class Exercise02:
    # 0 - left, 1 - up, 2 - right, 3 - down
    _gene_space = [0, 1, 2, 3]
    _sol_per_pop = 50
    _num_genes = 30
    _num_parents_mating = 25
    _num_generations = 100
    _keep_parents = 5
    _parent_selection_type = "sss"
    _crossover_type = "single_point"
    _mutation_type = "random"
    _mutation_percent_genes = 4
    _fitness_function = fitness_func
    _stop_criteria = "reach_0"
    _ga_instance = pygad.GA(
        gene_space=_gene_space,
        sol_per_pop=_sol_per_pop,
        num_genes=_num_genes,
        num_parents_mating=_num_parents_mating,
        num_generations=_num_generations,
        keep_parents=_keep_parents,
        parent_selection_type=_parent_selection_type,
        crossover_type=_crossover_type,
        mutation_type=_mutation_type,
        mutation_percent_genes=_mutation_percent_genes,
        fitness_func=_fitness_function
    )

    def run(self):
        start = time.time()

        self._ga_instance.run()

        end = time.time()

        solution, solution_fitness, solution_idx = self._ga_instance.best_solution()
        # print("Parameters of the best solution : {solution}".format(solution=solution))
        # print("Fitness value of the best solution = {solution_fitness}".format(
        # solution_fitness=solution_fitness))
        # print(f"Running time of the algorithm: {end - start}")

        return solution_fitness, solution, end - start
