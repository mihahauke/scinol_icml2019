#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################## LICENSE NOTE: #######################
# The MIT License (MIT)
#
# Copyright (c) 2016 Justin Johnson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##################################################################
import argparse
import json
import six
import numpy as np
import h5py
import codecs


def load_text(input_txt,
              encoding="utf-8",
              test_frac=0.1,
              verbose=False):
    # First go the file once to see how big it is and to build the vocab
    total_size = 0
    chars = set()
    blacklist = set(['\ufeff'])
    with codecs.open(input_txt, 'r', encoding) as f:
        for line in f:
            total_size += len(line)
            chars.update(line)
    # for token in blacklist:
    #     chars.remove(token)
    token_to_idx = {k: idx for idx, k in enumerate(sorted(chars))}
    # Now we can figure out the split sizes
    test_size = int(test_frac * total_size)
    train_size = total_size - test_size

    if verbose:
        print('Total vocabulary size: %d' % len(token_to_idx))
        print('Total tokens in file: %d' % total_size)
        print('  Training size: %d' % train_size)
        print('  Test size: %d' % test_size)

    # Choose the datatype based on the vocabulary size
    dtype = np.uint8
    if len(token_to_idx) > 255:
        dtype = np.uint32
    if verbose:
        print('Using dtype ', dtype)

    # Just load data into memory ... we'll have to do something more clever
    # for huge datasets but this should be fine for now
    train = np.zeros(train_size, dtype=dtype)
    test = np.zeros(test_size, dtype=dtype)
    splits = [train, test]

    # Go through the file again and write data to numpy arrays
    split_idx, cur_idx = 0, 0
    with codecs.open(input_txt, 'r', encoding) as f:
        for line in f:
            for char in line:
                if char not in blacklist:
                    splits[split_idx][cur_idx] = token_to_idx[char]
                    cur_idx += 1
                    if cur_idx == splits[split_idx].size:
                        split_idx += 1
                        cur_idx = 0
    return train, test, token_to_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_txt')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--encoding', '-e', default='utf-8')
    args = parser.parse_args()

    if args.encoding == 'bytes':
        args.encoding = None

    train, test, token_to_idx = load_text(
        input_txt=args.input_txt,
        encoding=args.encoding,
        test_frac=args.test_frac,
        verbose=args.verbose)

    idx_to_token = {v: k for k, v in token_to_idx.items()}
    # Write data to HDF5 file
    if args.output is None:
        pass
    else:
        output_h5 = args.output_ + ".h5"
        output_json = args.output + ".json"

        with h5py.File(output_h5, 'w') as f:
            f.create_dataset('train', data=train)
            f.create_dataset('test', data=test)

        # For 'bytes' encoding, replace non-ascii characters so the json dump
        # doesn't crash
        if args.encoding is None:
            new_token_to_idx = {}
            for token, idx in six.iteritems(token_to_idx):
                if ord(token) > 127:
                    new_token_to_idx['[%d]' % ord(token)] = idx
                else:
                    new_token_to_idx[token] = idx
            token_to_idx = new_token_to_idx

        # Dump a JSON file for the vocab
        json_data = {
            'token_to_idx': token_to_idx,
            'idx_to_token': {v: k for k, v in token_to_idx.items()},
        }
        with open(output_json, 'w') as f:
            json.dump(json_data, f)
