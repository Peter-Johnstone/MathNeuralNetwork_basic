# Goal: Extending from two to x weights, such that all the weights are stored in a one d numpy vector

import random
import numpy as np

RATE_OF_LEARNING = .0001


def get_answer(inputs):
    return .02*inputs[0] + -239*inputs[1] + -9*inputs[3] + 87.43*inputs[4] + 2983.16*inputs[5] + 43.2


### Structure: input - > x - > 1 hidden layer -> output

def forward_propagate(inputs: np.array, weights, bias):
    return np.dot(weights, inputs) + bias


def backward_propagate(inputs, weights, bias):
    solution = get_answer(inputs)
    guess = forward_propagate(inputs, weights, bias)
    difference = solution - guess

    # Adjust W:
    weights += RATE_OF_LEARNING * difference * inputs
    bias += RATE_OF_LEARNING * difference * 100
    return weights, bias


weights = np.random.randn(10)
bias = random.randint(-50, 50)
for i in range(10000):
    inputs = np.random.uniform(-50, 50, size=10)
    weights, bias = backward_propagate(inputs, weights, bias)
    if i % 100 == 0:
        print(weights[0:5])

print()
print(weights)



