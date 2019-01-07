#!/usr/bin/env python3

from util_plot import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import warnings
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import glob
import numpy as np
import os
import itertools as it
import traceback
import pandas as pd

# plt.style.use("ggplot")
from tqdm import tqdm

yellow = (0.86, 0.7612000000000001, 0.33999999999999997)
reddish = (0.86, 0.3712, 0.33999999999999997)
red_orange = "#e74c3c"
orange = (1.0, 0.4980392156862745, 0.054901960784313725)

green = (0.5688000000000001, 0.86, 0.33999999999999997)
blue = "#3498db"
violet = (0.6311999999999998, 0.33999999999999997, 0.86)
grey = (0.5, 0.5, 0.5)
black = "black"
white = (1.0, 1.0, 1.0)
light_gray = (0.95, 0.95, 0.95)
navy_blue = (0,0,0.5)

plt.grid(color=light_gray, which="both")

def set_ax_props(ax):
    pass
    # ax.set_facecolor(white)


names_dict = {
    "adadelta": "AdaDelta",
    "rmsprop": "rmsprop",
    "adam": "Adam",
    "adagrad": "AdaGrad",
    "sgd": "SGD",
    "scinol2": "ScInOL 2",
    "cocob": "CoCob",
}
colors_dict = {
    "adagrad": orange,
    "adam": black,
    "scinol2": blue,
    "sgd_dsqrt": grey,
    "cocob": red_orange,
    'sgd': grey,
    "rmsprop":violet,
    "adadelta":navy_blue

}

titles_dict = {
    "mnist": "MNIST",
    "UCI_Bank": "UCI Bank",
    "UCI_Covertype": "UCI Covertype",
    "UCI_Census": "UCI Census",
    "UCI_Madelon": "UCI Madelon",
    "Penn_shuttle": "Shuttle"
}
markers_dict = {
    "adagrad": "p",
    "adadelta": "*",
    "rmsprop": "x",
    "adam": "P",
    "scinol2": "v",
    "ScInOL_2": "v",
    "sgd": "d",
    "cocob": "X",

}

def plot_with_std(tree,
                  tag_sets,
                  title="TODO",
                  y_axis="TODO",
                  verbose=False,
                  x_axis_label="# minibatches",
                  line=None,
                  **kwargs):
    data = []
    steps = None
    min_runs_num = np.inf
    min_steps_num = np.inf
    max_runs_num = 0
    max_steps_num = 0
    for tags in tag_sets:
        series = tree.get(tags)
        steps = series[0, 0, :]
        values = series[:, 1, :]
        data.append(values)
        min_runs_num = min(min_runs_num, values.shape[0])
        min_steps_num = min(min_steps_num, values.shape[1])
        max_runs_num = max(max_runs_num, values.shape[0])
        max_steps_num = max(max_steps_num, values.shape[1])

    if verbose:
        if min_steps_num != max_steps_num:
            # TODO
            pass
        if min_runs_num != max_runs_num:
            # TODO
            pass

    steps = steps[0:min_steps_num]
    # TODO it's a workaround for different number of runs and points
    if len(data) > 1:
        data = [d[0:min_runs_num, 0:min_steps_num] for d in data]
        data = np.stack(data, axis=2)
    else:
        data = data[0]
        data = data.reshape(list(data.shape) + [1])

    short_names = [t[-1] for t in tag_sets]
    new_short_names = []
    for name in short_names:
        for group_name in sorted(names_dict, reverse=True):
            if name.startswith(group_name):
                new_short_names.append(group_name)
                break
    short_names = new_short_names

    default_kwargs = {
        "err_style": "ci_band",
        "ci": "sd",
    }
    default_kwargs.update(kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        ax = None
        for di in range(data.shape[2]):
            sn = short_names[di]
            ax = sns.tsplot(data[:, :, di],
                            time=steps,
                            value=y_axis,
                            condition=names_dict[sn],
                            legend=len(data.shape) > 2,
                            linewidth=1,
                            marker=markers_dict[sn],
                            color=colors_dict[sn],
                            markersize=4,
                            ax=ax,
                            **default_kwargs
                            )
            set_ax_props(ax)

        if line is not None:
            sns.tsplot([line] * len(steps),
                       color=black, time=steps, linewidth=1, linestyle='--', ax=ax,
                       )
    plt.title(title)
    plt.xlabel(x_axis_label)


def plot_with_std_v2(tree,
                     tag_sets,
                     title="TODO",
                     y_axis="TODO",
                     verbose=False,
                     x_axis_label="# iterations",
                     **kwargs):
    data = []
    steps = None
    min_runs_num = np.inf
    min_steps_num = np.inf
    max_runs_num = 0
    max_steps_num = 0
    for tags in tag_sets:
        series = tree.get(tags)
        steps = series[0, 0, :]
        values = series[:, 1, :]
        data.append(values)
        min_runs_num = min(min_runs_num, values.shape[0])
        min_steps_num = min(min_steps_num, values.shape[1])
        max_runs_num = max(max_runs_num, values.shape[0])
        max_steps_num = max(max_steps_num, values.shape[1])

    if verbose:
        if min_steps_num != max_steps_num:
            # TODO
            pass
        if min_runs_num != max_runs_num:
            # TODO
            pass

    steps = steps[0:min_steps_num]
    # TODO it's a workaround for different number of runs and points
    if len(data) > 1:
        data = [d[0:min_runs_num, 0:min_steps_num] for d in data]
        data = np.stack(data, axis=2)
        tag_sets = [t[-1] for t in tag_sets]
    else:
        tag_sets = None
        data = data[0]
    default_kwargs = {
        "err_style": "ci_band",
        "ci": "sd",
    }
    default_kwargs.update(kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Group tags:
        tag_groups = defaultdict(list)
        for i, tag in enumerate(tag_sets):
            for tag_group_key in names_dict:
                if tag.startswith(tag_group_key):
                    tag_groups[tag_group_key].append(i)

        ax = None
        for tag, indices in tag_groups.items():
            for i, di in enumerate(indices):
                ax = sns.tsplot(data[:, :, di],
                                time=steps,
                                # value=y_axis,
                                condition=names_dict[tag],
                                legend=i == 0,
                                ax=ax,
                                color=colors_dict[tag],
                                **default_kwargs
                                )
                set_ax_props(ax)
    plt.title(title)
    plt.xlabel(x_axis_label)


if __name__ == "__main__":
    parser = ArgumentParser("Plots multiple runs of benchmark algorithms.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--output-dir",
                        dest="output_dir",
                        default="graphs_b128")
    parser.add_argument("--log_dir",
                        default="tb_logs_linear_b128")
    parser.add_argument("--extension", "-e",
                        default="png")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        default=False)
    parser.add_argument("--list", "-l", action="store_true")

    parser.add_argument("--show", "-s", default=False, action="store_true")

    parser.add_argument("--log-scale", "-log", default=False, action="store_true")
    args = parser.parse_args()





    tree = Tree(verbose=args.verbose)

    filters = ["scinol2", "cocob", "adam", "adagrad", "adadelta", "rmsprop", "sgd"]
    # filters = ["scinol","scinol2", "prescinol"]
    excludes = ["prescinol2"]
    tree.load(args.log_dir, filters, excludes)

    if args.list:
        tree.print()
        exit(0)


    all_keys = list(tree.random_access_data.keys())

    # Plots everything separately
    print("Plotting separate graphs. Saving to: '{}'".format(args.output_dir))
    if not args.show:
        for key in tqdm(all_keys, leave=False):
            dataset, mode, architecture, algo = key
            try:
                plot_with_std(tree,
                              tag_sets=[key],
                              y_axis="cross entropy",
                              title="{}: {}".format(dataset, algo),
                              err_style="unit_traces",
                              )
                path = os.path.join(args.output_dir, dataset, algo)
                if args.log_scale:
                    plt.yscale("log")
                    path += "_log"
                save_plot(path, extension=args.extension)
            except Exception as ex:
                print("Failed for: {}".format(key))
                print("Data shape: {}".format(tree.get(key).shape))
                if args.verbose:
                    print(" ============== EXCEPTION ============")
                    print(ex)
                    traceback.print_exc()
                    print(" =====================================")
                    print()

    # Plot version 1
    print("Plotting joined graphs.")
    joined_keys = {}
    for d, m, a, algo in all_keys:
        if not (d, m, a) in joined_keys:
            joined_keys[(d, m, a)] = []
        joined_keys[(d, m, a)].append([d, m, a, algo])

    best_runs = {
        "mnist":
            {"adagrad_l0.1",
             "adadelta_l1.0",
             "rmsprop_l0.0005",
             "adam_l0.0001",
             "sgd_l1.0"},
        "UCI_Covertype":
            {"adagrad_l0.005",
             "adadelta_l0.1",
             "rmsprop_l1e-05",
             "adam_l0.0001",
             "sgd_l0.0001"},
        "UCI_Census":
            {"adagrad_l0.001",
             "adadelta_l0.5",
             "adam_l0.0001",
             "rmsprop_l1e-05",
             "sgd_l0.0001"},
        "UCI_Madelon":
            {"adagrad_l1e-05",
             "adadelta_l0.0005",
             "rmsprop_l0.0001",
             "adam_l0.0001",
             "sgd_l1.0"},
        "UCI_Bank":
            {"adagrad_l0.01",
             "adadelta_l0.1",
             "adam_l0.0005",
             "rmsprop_l0.0001",
             "sgd_l1e-05"},
        "Penn_shuttle":
            {"adagrad_l0.01",
             "adadelta_l0.5",
             "adam_l0.0001",
             "rmsprop_l0.0001",
             "sgd_l0.005"}
    }
    for best_set in best_runs.values():
        best_set.add("scinol")
        best_set.add("scinol2")
        best_set.add("cocob")

    for [d, m, a], tag_set in tqdm(joined_keys.items(), leave=False):
        new_tag_set = []
        for tags in tag_set:
            if tags[-1] in best_runs[d]:
                new_tag_set.append(tags)
        tag_set = sorted(new_tag_set, key=lambda x: x[3])
        plot_with_std(tree,
                      tag_sets=tag_set,
                      y_axis="cross entropy",
                      title=titles_dict[d],
                      line=None
                      )
        path = os.path.join(args.output_dir, d)
        if args.log_scale:
            plt.yscale("log")
            path += "_log"

        if args.show:
            plt.show()
        else:
            save_plot(path, extension=args.extension)

            # # Plot version 2
            # for [d, m, a], tag_set in tqdm(joined_keys.items(), leave=False):
            #     tag_set = sorted(tag_set, key=lambda x: x[3])
            #     plot_with_std_v2(tree,
            #                      tag_sets=tag_set,
            #                      y_axis="cross entropy",
            #                      title=titles_dict[d])
            #     save_plot(os.path.join(args.output_dir, "all_v2", d) + "_v2",
            #               extension=args.extension)
