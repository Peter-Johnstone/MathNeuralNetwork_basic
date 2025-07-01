# Goal: implement batch training


import random
import numpy as np

RATE_OF_LEARNING = .0001

def get_answer(inputs):
    return .02*inputs[0] + -239*inputs[1] + -9*inputs[2] + 87.43*inputs[3] + 2983.16*inputs[4] + 43.2


### Structure: input - > x - > 1 hidden layer -> output

def forward_propagate(inputs: np.array, weights, bias):
    return np.dot(weights, inputs) + bias


def backward_propagate(inputs, weights, bias):
    solution = get_answer(inputs)
    guess = forward_propagate(inputs, weights, bias)
    difference = solution - guess

    # Adjust W:
    gradients = RATE_OF_LEARNING * difference * inputs
    bias_difference = RATE_OF_LEARNING * difference * 100
    return gradients, bias_difference


def batch_train(batch_size, weights, bias):
    gradient_sum = 0
    bias_difference_sum = 0
    for _ in range(batch_size):
        inputs = np.random.uniform(-50, 50, size=10)
        gradient, bias_difference = backward_propagate(inputs, weights, bias)
        gradient_sum += gradient
        bias_difference_sum += bias_difference

    return weights + gradient_sum/batch_size, bias + bias_difference_sum/batch_size

# weights = np.random.randn(10)
# bias = random.randint(-50, 50)
# for i in range(10000):
#
#     weights, bias = batch_train(100, weights, bias)
#
#     if i % 100 == 0:
#         print(i, weights[0:5])
#
# print()
# print(weights[5:])



