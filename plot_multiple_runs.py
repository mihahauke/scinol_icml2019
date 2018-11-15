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

    def load(self, files, filters, excludes):
        if self.verbose:
            print("Loading files into tree structure")
            print("Found {} files...".format(len(files)))
        if self.verbose:
            files = tqdm(files)
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

            data = [steps, entropy]
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
        # if not isinstance(keys, list):
        #     keys = [keys]
        # elif tuple(keys) in self.random_access_data:
        #     return self.random_access_data[tuple(keys)]
        #
        # current_out = self.root
        # for k in keys:
        #     while k not in current_out and len(current_out) == 1:
        #         current_out = current_out[list(current_out.keys())[0]]
        #     current_out = current_out[k]
        # return current_out

    def _index(self):
        pass

    def _convert_lists_to_arrays(self, root=None):
        # parse lists to ndarrays
        # if root is None:
        #     root = self.root
        # for childname, child in root.items():
        #     if isinstance(child, dict):
        #         self._convert_lists_to_arrays(child)
        #     elif isinstance(child, list):
        #         max_len = max([len(series[0]) for series in child])
        #         child = [series for series in child if len(series[0]) == max_len]
        #         root[childname] = np.array(child, dtype=np.float32)
        #         self.random_access_data[tuple(childname)] = root[childname]
        #     else:
        #         raise ValueError("Defaq, should be list or dict, is: {}".format(type(child)))
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

    parser.add_argument("-o", "--output-dir", dest="output_dir", default="graphs")
    parser.add_argument("--log_dir", default="tb_logs")
    parser.add_argument("-i", "--interactive", action="store_true", help="TODO")
    parser.add_argument("-f", "--filters", nargs="*", default=[], help="TODO")
    parser.add_argument("-x", "--exclude", nargs="*", default=[], help="TODO")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--extension", default="pdf")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    all_files = glob.glob('{}/**/*events*'.format(args.log_dir), recursive=True)
    tree = Tree(verbose=args.verbose)

    tree.load(all_files, args.filters, args.exclude)

    if args.list:
        tree.print()
        exit(0)
    if args.interactive:
        raise NotImplementedError()

    # Plot everything separately
    all_keys = list(tree.random_access_data.keys())
    print("Saving graphs to: '{}'".format(args.output_dir))

    for key in tqdm(all_keys, leave=False):
        dataset, mode, architecture, algo = key
        try:
            plot_with_std(tree,
                          tag_sets=[key],
                          y_axis="cross entropy",
                          title="{}: {}".format(dataset, algo),
                          err_style="unit_traces")

            save_plot(os.path.join(args.output_dir, dataset, mode, architecture, algo),
                      extension=args.extension)
        except Exception as ex:
            print("Failed for: {}".format(key))
            print("Data shape: {}".format(tree.get(key).shape))
            if args.verbose:
                print(" ============== EXCEPTION ============")
                print(ex)
                traceback.print_exc()
                print(" =====================================")
            exit(0)
    joined_keys = {}
    # TODO it's ugly af ... refactor it ....
    # TODO and defaq does it dooo XD?
    for d, m, a, algo in all_keys:
        if not (d, m, a) in joined_keys:
            joined_keys[(d, m, a)] = []
        joined_keys[(d, m, a)].append([d, m, a, algo])

    for [d, m, a], tag_set in tqdm(joined_keys.items(), leave=False):
        try:
            tag_set = sorted(tag_set, key=lambda x: x[3])
            plot_with_std(tree,
                          tag_sets=tag_set,
                          y_axis="cross entropy",
                          title="{} {} {}".format(d, m, a))
            save_plot(os.path.join(args.output_dir, "{}_{}".format(d, m), a),
                      extension=args.extension)
        except Exception as ex:
            print("Failed for: {} + tags".format([d, m, a]))
            if args.verbose:
                print(" ============== EXCEPTION ============")
                traceback.print_exc()
                print(" =====================================")
            exit(0)
