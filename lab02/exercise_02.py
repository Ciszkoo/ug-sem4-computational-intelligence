#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

with open(f"{os.path.dirname(os.path.abspath(__file__))}/times.txt", "r") as file:
    lines = file.readlines()
    times = []
    for i in lines:
        time = i[:-2]
        times.append(np.float64(time))

mean = np.mean(times)

print(f"Average of the running time of the algorithm: {mean}")
