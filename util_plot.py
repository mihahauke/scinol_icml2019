#!/usr/bin/env python3

from util_plot import *
import warnings
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import glob
import numpy as np
import os
import itertools as it

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

    def load(self, logdir, filters=None, excludes=None):
        files = glob.glob('{}/**/*events*'.format(logdir), recursive=True)
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
                        if token.startswith(orfilter):
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
            if dataset == "UCI_Madelon":
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

    def _index(self):
        pass

    def _convert_lists_to_arrays(self, root=None):
        for tokens, series in self.random_access_data.items():
            self.random_access_data[tuple(tokens)] = self._lists_to_array(series)
            leaf = self.root
            for token in tokens[0:-1]:
                leaf = leaf[token]
            leaf[tokens[-1]] = self.random_access_data[tuple(tokens)]


def save_plot(path, extension="pdf", logscale=False, verbose=False):
    if not extension.startswith("."):
        extension = "." + extension
    if not path.endswith(extension):
        path += extension

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not logscale:
        plt.locator_params(nbins=6)
    if verbose:
        print("Saving {}".format(path))
    plt.savefig(path)
    plt.clf()



