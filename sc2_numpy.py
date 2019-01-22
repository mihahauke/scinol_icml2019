#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")

from datasets import *
import numpy as np

np.set_printoptions(threshold=np.nan)

SMALL_NUMBER = 1e-15

from matplotlib import pyplot as plt

plt.style.use("ggplot")


class Scinol2():
    def __init__(self, vars, dtype=np.float32):
        self.vars = vars
        self.M = {v: np.zeros_like(val, dtype=dtype) +SMALL_NUMBER  for v, val in vars.items()}
        self.S2 = {v: np.zeros_like(val, dtype=dtype) for v, val in vars.items()}
        self.G = {v: np.zeros_like(val, dtype=dtype) for v, val in vars.items()}
        self.eta = {v: np.ones_like(val, dtype=dtype) for v, val in vars.items()}

    def update(self, v, x):
        self.M[v] = np.maximum(self.M[v], abs(x))
        sqrt = (self.S2[v] + self.M[v] ** 2) ** 0.5
        theta = self.G[v] / sqrt
        self.vars[v][:] = np.sign(theta) * np.clip(np.abs(theta), None, 1.0) / (
                2 * sqrt) * self.eta[v]

    def post_update(self, v, g):
        self.G[v] -= g
        self.S2[v] += g ** 2
        self.eta[v] -= np.matmul(g, self.vars[v])


def single_run(epochs):
    dataset = SynthReg()
    # dataset = UCI_CTScan()
    x_train, y_train = dataset.train
    # limit = 100
    # x_train=x_train[0:limit]
    # y_train=y_train[0:limit]
    x_test, y_test = dataset.test
    y_train = y_train.reshape((-1,))
    y_test = y_test.reshape((-1,))

    dtype = np.float32
    b = np.zeros([1], dtype=dtype)
    w = np.zeros(x_train.shape[1], dtype=dtype)
    optimizer = Scinol2({'weights': w, 'bias': b}, dtype=dtype)

    def loss_fn(pred, target):
        return np.abs(pred - target)

    def grad_fn(pred, target, x):
        # print(pred-target,np.sign(pred-target),x)
        # print(np.sign(pred - target) * x)
        # print()
        return np.sign(pred - target) * x

    def train_epoch():
        perm = np.random.permutation(len(x_train))

        losses = []
        for x, target in zip(x_train[perm], y_train[perm]):
            optimizer.update('weights', x)
            optimizer.update('bias', 1.0)
            y = np.matmul(x, w) + b

            loss = loss_fn(y, target)
            grad_w = grad_fn(y, target, x)
            grad_b = grad_fn(y, target, 1.0)

            optimizer.post_update('weights', grad_w)
            optimizer.post_update('bias', grad_b)
            # print("x: {:0.7f}, \tw:     {:0.7f},  \tb: {:0.7f}".format(x[0], w[0], b[0]))
            # print("y: {:0.7f}, \tgw:    {:0.7f}, \tgb: {:0.7f}".format(y[0], grad_w[0], grad_b[0]))
            # print("t: {:0.7f}  \tGw:    {:0.7f}, \tGb: {:0.7f}".format(target,optimizer.G["weights"][0], optimizer.G["bias"][0]))
            # print("l: {:0.7f}  \tSw:    {:0.7f}, \tSb: {:0.7f}".format(loss[0], optimizer.S2["weights"][0], optimizer.S2["bias"][0]))
            # print("            \tEta w: {:0.7f}, \tEb: {:0.7f}".format(optimizer.eta["weights"][0], optimizer.eta["bias"][0]))
            # print("=================================")
            if np.isnan(loss) or np.isinf(loss):
                print("Nan/inf detected. Aborting!")
                # np.savetxt("bad_data.txt",np.array(XS).reshape([-1]))
                # exit(0)
                return losses
            losses.append(loss)
        return losses

    def test():
        y = np.matmul(x_test, w) + b
        loss = np.mean(loss_fn(y, y_test))
        return loss

    train_losses = []
    test_losses = []
    for _ in range(epochs):
        some_train_losses = train_epoch()

        train_losses += some_train_losses

        if len(some_train_losses) != len(x_train):
            break
        test_loss = test()
        test_losses.append(test_loss)

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    return train_losses, test_losses


runs = 10
from tqdm import trange

all_train_losses = []
all_test_losses = []
epochs = 1
for i in trange(runs):
    train_losses, test_losses = single_run(epochs)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

for train_losses in all_train_losses:
    if max(train_losses) - min(train_losses) > 10000:
        plt.yscale("log")
        print("Log scale enabled.")
    plt.plot(train_losses)

plt.show()
