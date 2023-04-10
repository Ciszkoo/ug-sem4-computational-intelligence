import numpy as np


def f_activation(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(age, weight, height):
    hidden1 = age * (- 0.46122) + weight * 0.97314 + \
        height * (- 0.39203) + 0.80109
    hidden1_activated = f_activation(hidden1)
    hidden2 = age * 0.78548 + weight * 2.10584 + height * (- 0.57847) + 0.43529
    hidden2_activated = f_activation(hidden2)
    output = hidden1_activated * (- 0.81546) + \
        hidden2_activated * 1.03775 - 0.2368
    return output


print(forward_pass(25, 67, 180))
