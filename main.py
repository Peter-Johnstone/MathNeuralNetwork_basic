
# Goal: Use machine learning to guess y = 2x + 1

import random

RATE_OF_LEARNING = .01

def get_answer(x):
    return 4.12*x + 43.2

### Structure: input - > x - > 1 hidden layer -> output

def forward_propagate(weight, bias, x):
    return weight*x + bias

def backward_propagate(weight, bias, x):

    solution = get_answer(x)
    guess = forward_propagate(weight, bias, x)
    difference = solution - guess
    # Adjust W:
    weight += RATE_OF_LEARNING * difference * x
    bias += RATE_OF_LEARNING * difference * 100
    return weight, bias


weight = random.randint(-15, 15)
bias = random.randint(-15, 15)
for i in range(300):
    x = random.randint(1, 1000) / 100
    weight, bias = backward_propagate(weight, bias, x)
    if i % 10 == 0:
        print(
            f"Step {i}, weight = {weight:.4f}, bias = {bias:.4f}, guess({x:.2f}) = {forward_propagate(weight, bias, x):.2f}")






