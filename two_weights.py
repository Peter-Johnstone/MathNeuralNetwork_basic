
# Goal: using two weights, IE: using machine learning to guess Z = 4x + 3y + 7
import random

RATE_OF_LEARNING = .01

def get_answer(x, y):
    return 23491*x + -5*y + 43.2

### Structure: input - > x - > 1 hidden layer -> output

def forward_propagate(weight1, weight2, bias, x, y):
    return weight1*x + weight2*y + bias

def backward_propagate(weight1, weight2, bias, x, y):
    solution = get_answer(x, y)
    guess = forward_propagate(weight1, weight2, bias, x, y)
    difference = solution - guess
    
    # Adjust W:
    weight1 += RATE_OF_LEARNING * difference * x
    weight2 += RATE_OF_LEARNING * difference * y
    bias += RATE_OF_LEARNING * difference * 100
    return weight1, weight2, bias


weight1 = random.randint(-15, 15)
weight2 = random.randint(-15, 15)
bias = random.randint(-15, 15)

#
# for i in range(10000):
#
#     # interestingly, we don't need to randomize a new x and y each time, if we didn't want to. We can repeat using the same
#     # points over and over again. If we have only one point, we will train to the model to get the correct answer for that one
#     # point. However, as there are three variables, there are various ways to manipulate those three variables to end up at that point.
#     # We will end up at one of them, but this is not enough to create generalized weights that work for any value. In order to be sure that
#     # we converge to the correct weights and biases (not just answer for the specific value data point) we need three unique and non overlapping
#     # points for the three variables.
#     x = random.randint(1, 1000) / 100
#     y = random.randint(1, 1000) / 100
#     weight1, weight2, bias = backward_propagate(weight1, weight2, bias, x, y)
#     if i % 100 == 0:
#         print(
#             f"Step {i}, weight1 = {weight1:.15f}, weight2 = {weight2:.15f}, bias = {bias:.4f}, guess({x:.2f}, {y:.2f}) = {forward_propagate(weight1, weight2, bias, x, y):.2f}, answer = {get_answer(x, y)}")
#
#
#



