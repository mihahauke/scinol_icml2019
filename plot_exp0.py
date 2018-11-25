#!/usr/bin/env python3

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

plt.style.use("ggplot")
from tqdm import tqdm

names_dict = {
    "adam": "Adam",
    "adagrad": "AdaGrad",
    "sgd_dsqrt": "SGD",
    "prescinol": "SI-Cw",
    "prescinol2": "SI-Full",
    "nag": "NAG"
}
yellow = (0.86, 0.7612000000000001, 0.33999999999999997)
reddish = (0.86, 0.3712, 0.33999999999999997)
red_orange = "#e74c3c"
orange = (1.0, 0.4980392156862745, 0.054901960784313725)

green = (0.5688000000000001, 0.86, 0.33999999999999997)
blue = "#3498db"
violet = (0.6311999999999998, 0.33999999999999997, 0.86)
grey = (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)

colors_dict = {
    "adagrad": orange,
    "adam": reddish,

    "nag": violet,
    "prescinol": green,
    "prescinol2": blue,

    "sgd_dsqrt": grey,
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
    "adam": "P",
    "nag": "s",
    "prescinol": "^",
    "prescinol2": "v",
    "sgd_dsqrt": "d",
}


# TODO maybe it's a stupid idea, maybe I can do it with pandas?
# it seems that not really, different datasets and test/train sets will have different dimensionality
class Tree(object):
    def __init__(self, verbose=False):
        def recursive_defaultdict_factory():
            return defaultdict(recursive_defaultdict_factory)

        self.root = defaultdict(recursive_defaultdict_factory)
        self.random_access_data = defaultdict(list)
        self.datasets = set()
        self.modes = set()
        self.architectures = set()
        self.algorithms = set()
        self.verbose = verbose

    def load(self, files, filters=None, excludes=None):
        if self.verbose:
            print("Loading files into tree structure")
            print("Found {} files...".format(len(files)))
        if self.verbose:
            files = tqdm(files)
        for filename in files:
            tokens = [x.strip("_") for x in filename.strip().split("/")]
            stop = False
            if filters is not None and len(filters) > 0:
                stop = True
                for orfilter in filters:
                    for token in tokens:
                        if orfilter in token:
                            stop = False
                            break
            if excludes is not None and len(excludes) > 0:
                for ex, t in it.product(excludes, tokens):
                    if ex in t:
                        stop = True
                        break
            if stop:
                continue

            dataset = tokens[1]
            mode = tokens[5]
            architecture = tokens[2]
            algo = tokens[4]
            self.datasets.add(dataset)
            self.modes.add(mode)
            self.architectures.add(architecture)
            self.algorithms.add(algo)

            tokens_list = [dataset, mode, architecture, algo]
            acc = []
            entropy = []
            steps = []
            try:
                for event in tf.train.summary_iterator(filename):
                    if event.HasField('summary'):
                        steps.append(event.step)
                        for value in event.summary.value:
                            if value.tag.endswith("/accuracy"):
                                acc.append(value.simple_value)
                            elif value.tag.endswith("/cross_entropy"):
                                entropy.append(value.simple_value)
            except:
                print("Could not read '{}'".format(filename))

            data = [steps[1:], entropy[1:]]

            self._add_leaf(tokens_list, data)

        self._convert_lists_to_arrays()
        self._index()

    def _lists_to_array(self, the_list, filler=None):
        max_len = max([len(sublist[0]) for sublist in the_list])
        if filler is None:
            the_list = [sublist for sublist in the_list if len(sublist[0]) == max_len]
        else:
            for t, values in the_list:
                addition = [filler] * (max_len - len(t))
                t += addition
                values += addition

        return np.array(the_list, dtype=np.float32)

    def _add_leaf(self, tokens, series):
        assert len(series) == 2
        assert len(series[0]) == len(series[1])

        if self.verbose:
            if len(series[0]) == 1:
                print("WARNING: series len=1 for tokens: {}. Omitting.".format(tokens))
                return

        current_dict = self.root
        for t in tokens:
            current_dict = current_dict[t]
        key = tuple(tokens)
        self.random_access_data[key].append(series)

    def _print_recursive(self, item, indent_level=0):
        if isinstance(item, dict):
            for x in sorted(item):
                print("{}{}:".format("  " * indent_level, x))
                self._print_recursive(item[x], indent_level + 1)
        else:
            print("{}{}x{} series".format("  " * indent_level, item.shape[0], item.shape[2]))
            # print("{}{} series:".format("  " * (indent_level - 1), len(item)))
            # for series in item:
            #     # TODO well. . . its padded anywaaaay . . .
            #     print("{}{} points".format("  " * indent_level, len(series[0])))

    def print(self):
        self._print_recursive(self.root)

    def print_flat(self):
        raise NotImplementedError()

    def get(self, keys):
        return self.random_access_data[tuple(keys)]

    def _index(self):
        pass

    def _convert_lists_to_arrays(self, root=None):
        for tokens, series in self.random_access_data.items():
            self.random_access_data[tuple(tokens)] = self._lists_to_array(series)
            leaf = self.root
            for token in tokens[0:-1]:
                leaf = leaf[token]
            leaf[tokens[-1]] = self.random_access_data[tuple(tokens)]


def plot_with_std(tree,
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
        ax = None
        for di in range(data.shape[2]):
            sn = short_names[di]
            sns.tsplot(data[:, :, di],
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
    plt.title(title)
    plt.xlabel(x_axis_label)


def save_plot(path, extension="pdf"):
    if not extension.startswith("."):
        extension = "." + extension
    if not path.endswith(extension):
        path += extension

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.locator_params(nbins=6)
    plt.savefig(path)
    plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser("Plots multiple runs of benchmark algorithms.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--output-dir",
                        dest="output_dir",
                        default="graphs")
    parser.add_argument("--log_dir",
                        default="tb_logs")
    parser.add_argument("--extension", "-e",
                        default="png")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    all_files = glob.glob('{}/**/*events*'.format(args.log_dir), recursive=True)
    tree = Tree(verbose=args.verbose)

    filters = ["adam", "adagrad", "prescinol", "nag", "sgd_dsqrt", "prescinol2"]
    tree.load(all_files, filters)

    # Plots everything separately
    all_keys = list(tree.random_access_data.keys())
    print("Saving graphs to: '{}'".format(args.output_dir))

    for key in tqdm(all_keys, leave=False):
        dataset, mode, architecture, algo = key
        try:
            plot_with_std(tree,
                          tag_sets=[key],
                          y_axis="cross entropy",
                          title="{}: {}".format(dataset, algo),
                          err_style="unit_traces",
                          )

            save_plot(os.path.join(args.output_dir, dataset, algo),
                      extension=args.extension)
        except Exception as ex:
            print("Failed for: {}".format(key))
            print("Data shape: {}".format(tree.get(key).shape))
            if args.verbose:
                print(" ============== EXCEPTION ============")
                print(ex)
                traceback.print_exc()
                print(" =====================================")

    joined_keys = {}
    for d, m, a, algo in all_keys:
        if not (d, m, a) in joined_keys:
            joined_keys[(d, m, a)] = []
        joined_keys[(d, m, a)].append([d, m, a, algo])

    # Plot version 1

    best_runs = {
        "mnist":
            {"adagrad_l0.1",
             "adam_l0.0001",
             "nag_l1.0",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l1.0"},
        "UCI_Covertype":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l1.0",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l1e-05"},
        "UCI_Census":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l0.01",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l1e-05"},

        "UCI_Madelon":
            {"adagrad_l0.0001",
             "adam_l0.0001",
             "nag_l0.1",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l1e-05"},

        "UCI_Bank":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l1.0",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l1e-05"},
        "Penn_shuttle":
            {"adagrad_l0.01",
             "adam_l0.0001",
             "nag_l0.1",
             "prescinol2",
             "prescinol",
             "sgd_dsqrt_l0.0001"}
    }

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
                      )
        save_plot(os.path.join(args.output_dir, d),
                  extension=args.extension)

        # # Plot version 2
        # for [d, m, a], tag_set in tqdm(joined_keys.items(), leave=False):
        #     tag_set = sorted(tag_set, key=lambda x: x[3])
        #     plot_with_std_v2(tree,
        #                      tag_sets=tag_set,
        #                      y_axis="cross entropy",
        #                      title=titles_dict[d])
        #     save_plot(os.path.join(args.output_dir, "all_v2", d) + "_v2",
        #               extension=args.extension)
