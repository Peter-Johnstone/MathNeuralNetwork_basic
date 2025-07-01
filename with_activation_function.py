# Simple
import random

import numpy as np

RATE_OF_LEARNING = .1


def get_answer(x):
    return 10*sigmoid(3*x-1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def get_initial_weights_and_biases():
    w1 = random.randint(-50, 50)/50
    b1 = random.randint(-50, 50)/50
    w2 = random.randint(-50, 50)/50
    b2 = random.randint(-50, 50)/50
    return w1, b1, w2, b2

def forward_propagate(x, w1, b1, w2, b2):
    z1 = x*w1 + b1
    a1 = sigmoid(z1)
    z2 = a1*w2 + b2

    # a2 = sigmoid(z2)
    # return a2, z2, a1, z1
    return z2, a1, z1

def backward_propagate(x, w1, b1, w2, b2):
    y = get_answer(x)
    a2, a1, z1 = forward_propagate(x, w1, b1, w2, b2)

    # second layer
    overlap2 = 2 * (a2 - y)
    dw2 = overlap2 * a1
    db2 = overlap2

    # first layer
    overlap1 = d_sigmoid(z1) * w2 * overlap2
    dw1 = overlap1 * x
    db1 = overlap1

    return dw1, db1, dw2, db2

def run():
    w1, b1, w2, b2 = get_initial_weights_and_biases()
    for i in range(1000000):
        x = random.randint(-50, 50)/50
        dw1, db1, dw2, db2 = backward_propagate(x, w1, b1, w2, b2)
        w1 = w1 - RATE_OF_LEARNING * dw1
        b1 = b1 - RATE_OF_LEARNING * db1
        w2 = w2 - RATE_OF_LEARNING * dw2
        b2 = b2 - RATE_OF_LEARNING * db2

        if i % 100 == 0:
            print("Iteration:", i, "Guess:", forward_propagate(x, w1, b1, w2, b2)[0], "Actual:", get_answer(x))


if __name__ == '__main__':
    run()