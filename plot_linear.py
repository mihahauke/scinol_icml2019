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

names_dict = {
    "adam": "Adam",
    "Adam": "Adam",
    "adagrad": "AdaGrad",
    'AdaGrad': "AdaGrad",
    "sgd_dsqrt": "SGD",
    'SGD': "SGD",
    "sgd": "SGD",
    "scinol": "ScInOL 1",
    "scinol2": "ScInOL 2",
    "ScInOL_1": "ScInOL 1",
    "ScInOL_2": "ScInOL 2",
    "nag": "NAG",
    "cocob": "CoCob",
    "Orabona": "SFMD",
    'NAG': "NAG",
    'prescinol': 'Prescinol',
    'prescinol_ed': "Prescinol D",
    'prescinol_edt': "Alg1-K17"
}
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

axis_labels = {"cross_entropy": "cross entropy", "accuracy": "accuracy"}
file_suffixes =  {"cross_entropy": "ce", "accuracy": "acc"}

def set_ax_props(ax):
    plt.grid(color=light_gray, which="both")
    # ax.set_facecolor(white)


colors_dict = {
    "adagrad": orange,
    'AdaGrad': orange,
    "adam": black,
    "nag": violet,
    "scinol": green,
    "scinol2": blue,
    "ScInOL_1": green,
    "ScInOL_2": blue,
    "sgd_dsqrt": grey,
    "Adam": black,
    "cocob": red_orange,
    "Orabona": red_orange,
    'NAG': violet,
    'SGD': grey,
    'prescinol': violet,
    'prescinol_ed': red_orange,
    'prescinol_edt': navy_blue

}

titles_dict = {
    "mnist": "MNIST",
    "UCI_Bank": "Bank (UCI)",
    "UCI_Covertype": "Covertype (UCI)",
    "UCI_Census": "Census (UCI)",
    "UCI_Madelon": "Madelon (UCI)",
    "Penn_shuttle": "Shuttle (UCI)"
}
classes = {
    "mnist": 10,
    "UCI_Bank": 2,
    "UCI_Covertype": 7,
    "UCI_Census": 2,
    "UCI_Madelon": 2,
    "Penn_shuttle": 7
}
markers_dict = {
    "adagrad": "p",
    'AdaGrad': "p",
    "adam": "P",
    "nag": "s",
    "scinol": "^",
    "scinol2": "v",
    "ScInOL_1": "^",
    "ScInOL_2": "v",
    "sgd_dsqrt": "d",
    "Adam": "P",

    "Orabona": "X",
    "cocob": "X",
    'NAG': "s",
    'SGD': "d",
    'prescinol': "^",
    'prescinol_ed': "v",
    'prescinol_edt': "*"

}


def plot_with_std(tree,
                  tag_sets,
                  title="TODO",
                  y_axis="TODO",
                  verbose=False,
                  x_axis_label="# iterations",
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
                        default="graphs_linear")
    parser.add_argument("--log_dir",
                        default="tb_logs_linear")
    parser.add_argument("--extension", "-e",
                        default="png")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        default=False)
    parser.add_argument("--list", "-l", action="store_true")

    parser.add_argument("--show", "-s", default=False, action="store_true")

    parser.add_argument("--log-scale", "-log", default=False, action="store_true")
    parser.add_argument("--key", "-k", default="cross_entropy")
    args = parser.parse_args()

    # artificial exp :
    df = pd.read_csv("artificial_new.csv")
    header = df.columns.values[1:48]
    data = df.values[:, 1:48]
    t = df.values[:, 0]
    t = np.append(t, [0])
    START_ENTROPY = np.log(2)
    BEST_ENTROPY = 0.26
    runs = defaultdict(list)
    for h, y in zip(header, data.T):
        name = h.split(" ")[0]
        y = np.append(y, [START_ENTROPY])
        runs[name].append(y)


    def plot():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            ax = None
            for name in runs:
                for i, run in enumerate(runs[name]):
                    ax = sns.tsplot(run,
                                    condition=names_dict[name],
                                    color=colors_dict[name],
                                    time=t,
                                    markersize=4,
                                    linewidth=1 / len(runs[name]),
                                    legend=(i == 0),
                                    value=axis_labels["cross_entropy"],
                                    ax=ax,
                                    )
            ax = sns.tsplot([BEST_ENTROPY] * len(t),
                            color=black, time=t, linewidth=1, linestyle='--', ax=ax,
                            )
            sns.tsplot([START_ENTROPY] * len(t),
                       color=black, time=t, linewidth=1, linestyle='--', ax=ax,
                       )
            set_ax_props(ax)
            plt.xlabel("# iterations")
            plt.xlabel("# iterations")


    tree = Tree(key=args.key,verbose=args.verbose)

    filters = ["scinol", "scinol2", "cocob", "adam", "adagrad", "nag", "sgd_dsqrt","prescinol_edt"]
    # filters = ["scinol","scinol2", "prescinol"]

    excludes = ["prescinol2"]
    tree.load(args.log_dir, filters, excludes)

    if args.list:
        tree.print()
        exit(0)

    print("Plotting artificial experiment...")
    plot()
    plt.yscale("log")
    plt.title("Artificial data")
    path = os.path.join(args.output_dir, "artificial")
    save_plot(path, extension=args.extension, logscale=True, verbose=args.verbose)

    plot()
    plt.title("Artificial data (zoom)")
    plt.ylim(BEST_ENTROPY, START_ENTROPY)
    path = os.path.join(args.output_dir, "artificial_zoom")
    save_plot(path, extension=args.extension, verbose=args.verbose)

    all_keys = list(tree.random_access_data.keys())

    # Plots everything separately
    print("Plotting separate graphs. Saving to: '{}'".format(args.output_dir))
    if not args.show:
        for key in tqdm(all_keys, leave=False):
            dataset, mode, architecture, algo = key
            try:
                plot_with_std(tree,
                              tag_sets=[key],
                              y_axis=axis_labels[args.key],
                              title="{}: {}".format(dataset, algo),
                              err_style="unit_traces",
                              )
                path = os.path.join(args.output_dir, dataset,file_suffixes[args.key], algo)
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
             "adam_l0.0001",
             "nag_l1.0",
             "sgd_dsqrt_l1.0"},
        "UCI_Covertype":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l1.0",
             "sgd_dsqrt_l1e-05"},
        "UCI_Census":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l0.01",
             "sgd_dsqrt_l1e-05"},

        "UCI_Madelon":
            {"adagrad_l0.0001",
             "adam_l0.0001",
             "nag_l0.1",
             "sgd_dsqrt_l1e-05"},

        "UCI_Bank":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l1.0",
             "sgd_dsqrt_l1e-05"},
        "Penn_shuttle":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l0.1",
             "sgd_dsqrt_l0.0001"}
    }
    for best_set in best_runs.values():
        best_set.add("scinol")
        best_set.add("scinol2")
        best_set.add("cocob")
        # best_set.add("prescinol")
        # best_set.add("prescinol_ed")
        best_set.add("prescinol_edt")

    for [d, m, a], tag_set in tqdm(joined_keys.items(), leave=False):
        new_tag_set = []
        for tags in tag_set:
            if tags[-1] in best_runs[d]:
                new_tag_set.append(tags)
        tag_set = sorted(new_tag_set, key=lambda x: x[3])
        plot_with_std(tree,
                      tag_sets=tag_set,
                      y_axis=axis_labels[args.key],
                      title=titles_dict[d],
                      line=None
                      )
        path = os.path.join(args.output_dir, file_suffixes[args.key],d)
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
