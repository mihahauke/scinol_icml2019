#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import warnings

from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import glob
import numpy as np
import os
import itertools as it

plt.style.use("ggplot")


# TODO maybe it's a stupid idea, maybe I can do it with pandas?
# it seems that not really, different datasets and test/train sets will have different dimensionality
class Tree(object):
    def __init__(self):
        self.root = dict()
        self.random_access_data = {}
        self.datasets = set()
        self.modes = set()
        self.architectures = set()
        self.algorithms = set()

    def load(self, files, filters, excludes):
        for filename in files:
            tokens = [x.strip("_") for x in filename.strip().split("/")]
            stop = False
            for filter in filters:
                if filter not in tokens:
                    stop = True
                    break
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
            for event in tf.train.summary_iterator(filename):
                if event.HasField('summary'):
                    steps.append(event.step)
                    for value in event.summary.value:
                        if value.tag.endswith("/accuracy"):
                            acc.append(value.simple_value)
                        elif value.tag.endswith("/cross_entropy"):
                            entropy.append(value.simple_value)

            data = [steps, entropy]
            key = tuple(tokens_list)
            if key not in self.random_access_data:
                self.random_access_data[key] = []
            self.random_access_data[key].append(data)

            tree.add(tokens_list, data)
        tree._clean_recursive()
        tree._index()

    def add(self, tokens, data):

        current_dict = self.root

        for t in tokens[0:-1]:
            if t not in current_dict:
                current_dict[t] = dict()
            current_dict = current_dict[t]
        if tokens[-1] not in current_dict:
            current_dict[tokens[-1]] = []
        last_level = current_dict[tokens[-1]]
        last_level.append(data)

    def _print_recursive(self, item, indent_level=0):
        if isinstance(item, dict):
            for x in sorted(item):
                print("{}{}:".format("  " * indent_level, x))
                self._print_recursive(item[x], indent_level + 1)
        else:
            print("{}{} samples".format("  " * indent_level, len(item)))
            # for x in item:
            #     print("{}{}".format("  " * indent_level, x.split("/")[-1]))

    def print(self):
        self._print_recursive(self.root)

    def get(self, keys):
        if not isinstance(keys, list):
            keys = [keys]
        elif tuple(keys) in self.random_access_data:
            return self.random_access_data[tuple(keys)]

        current_out = self.root
        for k in keys:
            while k not in current_out and len(current_out) == 1:
                current_out = current_out[list(current_out.keys())[0]]
            current_out = current_out[k]
        return current_out

    def _index(self):
        pass

    def _clean_recursive(self, item=None):
        # parse lists to ndarrays
        if item is None:
            item = self.root
        for x in item:
            if isinstance(item[x], dict):
                self._clean_recursive(item[x])
            elif isinstance(item[x], list):
                item[x] = np.float32(item[x])
                # TODO won't work if number of steps doesn't match - doesn't matter for now
        for x in self.random_access_data:
            self.random_access_data[x] = np.float32(self.random_access_data[x])


def plot_with_std(tree, tag_sets, title="TODO", y_axis="TODO", **kwargs):
    data = []
    steps = None
    min_runs_num = np.inf
    for tags in tag_sets:
        series = tree.get(tags)
        steps = series[0, 0, :]
        values = series[:, 1, :]
        data.append(values)
        min_runs_num = min(min_runs_num, values.shape[0])

    # TODO it's a workaround for different number of samples
    data = [d[0:min_runs_num] for d in data]

    if len(data) > 1:
        data = np.stack(data, axis=2)
        tag_sets = [t[-1] for t in tag_sets]
    else:
        tag_sets = None
        data = data[0]
    x_axis_label = "# of minibatches"

    default_kwargs = {
        "err_style": "ci_band",
        "ci": "sd",

    }
    default_kwargs.update(kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sns.tsplot(data,
                   time=steps,
                   value=y_axis,
                   condition=tag_sets,
                   legend=len(data.shape) > 2,
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

    plt.savefig(path)
    plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser("Plots multiple runs of benchmark algorithms.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--output-dir",dest="output_dir", default="graphs")
    parser.add_argument("--log_dir", default="tb_logs")
    parser.add_argument("-i", "--interactive", action="store_true", help="TODO")
    parser.add_argument("-f", "--filters", nargs="*", default=[], help="TODO")
    parser.add_argument("-x", "--exclude", nargs="*", default=[], help="TODO")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--extension", default="pdf")
    args = parser.parse_args()

    all_files = glob.glob('{}/**/*events*'.format(args.log_dir), recursive=True)

    tree = Tree()
    tree.load(all_files, args.filters, args.exclude)

    if args.list:
        tree.print()
        exit(0)
    if args.interactive:
        raise NotImplementedError()

    # Plot everything separately
    all_keys = tree.random_access_data

    print("Saving graphs to: '{}'".format(args.output_dir))
    for dataset, mode, architecture, algo in all_keys:
        plot_with_std(tree,
                      tag_sets=[[dataset, mode, architecture, algo]],
                      y_axis="cross entropy",
                      title="{}: {}".format(dataset, algo),
                      err_style="unit_traces")

        save_plot(os.path.join(args.output_dir, dataset, mode, architecture, algo),
                  extension=args.extension)

    joined_keys = {}
    # TODO it's ugly af ... refactor it ....
    for d, m, a, algo in all_keys:
        if not (d, m, a) in joined_keys:
            joined_keys[(d, m, a)] = []
        joined_keys[(d, m, a)].append([d, m, a, algo])

    for [d, m, a], tag_set in joined_keys.items():
        plot_with_std(tree,
                      tag_sets=tag_set,
                      y_axis="cross entropy",
                      title="{} {} {}".format(d, m, a))
        save_plot(os.path.join(args.output_dir, "{}_{}".format(d, m), a),
                  extension=args.extension)


