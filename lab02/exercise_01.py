#!/usr/bin/env python
# coding: utf-8

import pygad
import numpy as np
import time
import os

data = [
    {"nazwa": "zegar", "waga": 7, "wartosc": 100},
    {"nazwa": "obraz-pejzaz", "waga": 7, "wartosc": 300},
    {"nazwa": "obraz-portret", "waga": 6, "wartosc": 200},
    {"nazwa": "radio", "waga": 2, "wartosc": 40},
    {"nazwa": "laptop", "waga": 5, "wartosc": 500},
    {"nazwa": "lampka nocna", "waga": 6, "wartosc": 70},
    {"nazwa": "srebrne sztucce", "waga": 1, "wartosc": 100},
    {"nazwa": "porcelana", "waga": 3, "wartosc": 250},
    {"nazwa": "figurka z brazu", "waga": 10, "wartosc": 300},
    {"nazwa": "skorzana torebka", "waga": 3, "wartosc": 280},
    {"nazwa": "odkurzacz", "waga": 15, "wartosc": 300}
]

weight_arr = [item["waga"] for item in data]

value_arr = [item["wartosc"] for item in data]

gene_space = [0, 1]


def fitness_func(solution, solution_idx):
    weight_sum = np.sum(weight_arr * solution)
    if (weight_sum > 25):
        return 0
    value_sum = np.sum(value_arr * solution)
    return value_sum


fitness_function = fitness_func

sol_per_pop = 20
num_genes = len(data)

num_parents_mating = 10
num_generations = 50
keep_parents = 4

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

stop_criteria = "reach_1600"

for i in range(0, 10):

    start = time.time()

    ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria=stop_criteria)

    ga_instance.run()

    end = time.time()

    running_time = end - start

    with open(f"{os.path.dirname(os.path.abspath(__file__))}/times.txt", "a") as file:
        file.write(f"{running_time}\n")

print("Last iteration of loop:\n")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))

prediction = [item for index, item in enumerate(data) if solution[index] == 1]
print(f"Predicted items based on the best solution: {prediction}")

print(f"Generations passed: {ga_instance.generations_completed}")

print(f"Running time of algorithm: {running_time}")


ga_instance.plot_fitness()
