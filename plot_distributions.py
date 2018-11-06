#!/usr/bin/env python3

from distributions import normal_dist_outliers, normal_scaled
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


def plot_dist(x, labels, probs, jitter, alpha, name):
    print("Noise:",np.mean(np.minimum(probs, 1 - probs)))
    # plt.scatter(probs, labels, alpha=0.1)
    # plt.gcf().canvas.set_window_title(name + " probs/labels")
    # plt.yticks([0, 1])
    # plt.xlabel("probability")
    # plt.ylabel("label")
    # plt.show()

    plt.hist(probs, bins=20, normed=True)
    plt.xlabel("(probability dist)")
    plt.gcf().canvas.set_window_title(name + "probabilities")
    plt.show()

    sns.stripplot(data=x,
                  jitter=jitter,
                  alpha=alpha)
    plt.gcf().canvas.set_window_title(name + "(features)")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--size", default=10000, help="dataset size", type=int)
    parser.add_argument("-f", "--num-features", default=10, type=int)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-nj", "--no-jitter", dest="no_jitter", action="store_true")
    parser.add_argument("-a", "--alpha", default=0.5, type=float)

    args = parser.parse_args()
    def plot(dist_func,name, **kwargs):
        x, labels, probs = dist_func(
            size=args.size,
            num_features=args.num_features,
            loc=0,
            seed=args.seed,
            return_probs=True,
            **kwargs)
        plot_dist(x,
                  labels,
                  probs,
                  not args.no_jitter,
                  alpha=args.alpha,
                  name=name)

    plot(normal_dist_outliers,name="Normal with outliers")
    # plot(normal_scaled, name="Normal scaled 2^10", max_exponent=10)