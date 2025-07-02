# Simple
import random

import numpy as np
import matplotlib.pyplot as plt

RATE_OF_LEARNING = .001
HIDDEN_NEURONS = 32


def get_answer(x):
    return 10*sigmoid(3*x-1)

def ReLU(x):
    # works for arrays
    return np.maximum(0, x)

def d_ReLU(x):
    return x>0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def get_initial_weights_and_biases():
    # w1 = random.randint(-50, 50)/50
    w1 = np.random.randn(HIDDEN_NEURONS) *.1
    b1 = np.random.randn(HIDDEN_NEURONS) *.1
    w2 = np.random.randn(HIDDEN_NEURONS) *.1
    b2 = random.randint(-50, 50)/100
    return w1, b1, w2, b2

def forward_propagate(x, w1, b1, w2, b2):

    inputs = np.full(HIDDEN_NEURONS, x)
    z1 = inputs*w1 + b1
    a1 = ReLU(z1)
    z2 = np.dot(a1, w2) + b2

    return z2, a1, z1

def backward_propagate(x, w1, b1, w2, b2):
    y = get_answer(x)
    z2, a1, z1 = forward_propagate(x, w1, b1, w2, b2)

    # second layer
    overlap2 = 2 * (z2 - y)
    dw2 = overlap2 * a1
    db2 = overlap2

    # first layer
    overlap1 = d_ReLU(z1) * w2 * overlap2
    dw1 = overlap1 * x
    db1 = overlap1

    return dw1, db1, dw2, db2



def run():
    w1, b1, w2, b2 = get_initial_weights_and_biases()
    x_vals = np.linspace(-1, 2, 200)  # input range for plotting

    for i in range(10000000):
        x = random.uniform(-1, 2)
        dw1, db1, dw2, db2 = backward_propagate(x, w1, b1, w2, b2)
        w1 = w1 - RATE_OF_LEARNING * dw1
        b1 = b1 - RATE_OF_LEARNING * db1
        w2 = w2 - RATE_OF_LEARNING * dw2
        b2 = b2 - RATE_OF_LEARNING * db2

        # visualize every 1000 steps
        if i % 10000 == 0:
            y_preds = [forward_propagate(xi, w1, b1, w2, b2)[0] for xi in x_vals]
            y_true = [get_answer(xi) for xi in x_vals]

            plt.clf()
            plt.title(f"Step {i}")
            plt.plot(x_vals, y_true, label="Target", linewidth=2)
            plt.plot(x_vals, y_preds, label="Prediction", linestyle="--")
            plt.legend()
            plt.pause(0.01)

    plt.show()


if __name__ == '__main__':
    run()