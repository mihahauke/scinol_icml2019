#!/usr/bin/env python3


import argparse
from tqdm import trange
import ruamel.yaml as yaml
from time import strftime

import tensorflow as tf
from collections import defaultdict
from adalbert import *
from scinol import *
from cocob import *
import itertools as it
from models import *
from datasets import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DEFAULT_TIMES = 1
DEFAULT_TB_LOGDIR = "tb_logs"

FLUSH_SECS = 2
TESTBATCH_SIZE = 100000
DEFAULT_EPOCHS = 30
DEFAULT_ONE_HOT = True


def test(
        logdir,
        dataset,
        model,
        optimizer_class,
        optimizer_args,
        epochs=DEFAULT_EPOCHS,
        one_hot=DEFAULT_ONE_HOT,
        train_histograms=False,
        tag=None,
        *args,
        **kwargs):
    # TODO add tag support
    if tag is not None:
        raise NotImplementedError()

    tf.gfile.MakeDirs(logdir)
    tf.reset_default_graph()
    dropout_switch = tf.placeholder_with_default(1.0,
                                                 None,
                                                 name='dropout_switch')

    x = tf.placeholder(tf.float32, [None] + dataset.input_shape, name='x-input')
    y = eval(model)(x, dataset.outputs_num, dropout_switch=dropout_switch)

    if dataset.outputs_num == 1:
        y_target = tf.placeholder(tf.float32, [None], name='y-input')
        flat_y = tf.reshape(y, [-1])
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=flat_y)
        correct_predictions = tf.equal(tf.cast(tf.greater(flat_y, 0), tf.float32), y_target)
    else:
        if one_hot:
            y_target = tf.placeholder(tf.float32, [None, dataset.outputs_num], name='y-input')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=y)
            correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
        else:
            y_target = tf.placeholder(tf.int64, [None], name='y-input')
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y)
            correct_predictions = tf.equal(tf.argmax(y, 1), y_target)

    mean_cross_entropy = tf.reduce_mean(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    # accuracy = tf.metrics.accuracy(y_target, tf.reshape(y,[-1]))

    loss = mean_cross_entropy
    optimizer = eval(optimizer_class)(**optimizer_args)
    # TODO workaround
    pretrain_step = None
    preminimize_op = getattr(optimizer, "pre_minimize", None)
    if callable(preminimize_op):
        pretrain_step = preminimize_op(x)

    grads_and_vars = optimizer.compute_gradients(loss)
    train_step = optimizer.apply_gradients(grads_and_vars)

    summaries_prefix = dataset.get_name()
    hist_summaries = []
    for grad, var in grads_and_vars:
        g_summary = tf.summary.histogram('{}/{}/gradients/{}'.format(summaries_prefix, model, var.name), grad)
        v_summary = tf.summary.histogram('{}/{}/{}'.format(summaries_prefix, model, var.name), var)
        hist_summaries.append(g_summary)
        hist_summaries.append(v_summary)
    acc_summary = tf.summary.scalar('{}/accuracy'.format(summaries_prefix), accuracy)
    ce_summary = tf.summary.scalar('{}/cross_entropy'.format(summaries_prefix), mean_cross_entropy)

    all_summaries = tf.summary.merge_all()
    # all_summaries=  tf.summary.merge([acc_summary, ce_summary])
    if train_histograms:
        train_summaries = all_summaries
    else:
        train_summaries = tf.summary.merge([acc_summary, ce_summary])

    time = strftime("%m.%d_%H-%M-%S")
    optim_name = optimizer.get_name().lower()
    oargs = "_".join(k[0] + str(v) for k, v in sorted(optimizer_args.items()))
    prefix = "{}/{}/{}/{}/{}_{}".format(logdir, dataset.get_name(), model, time, optim_name, oargs)
    prefix = prefix.strip("_")
    train_writer = tf.summary.FileWriter(prefix + '/train', flush_secs=FLUSH_SECS)
    test_writer = tf.summary.FileWriter(prefix + '/test', flush_secs=FLUSH_SECS)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batches_processed = 0

    test_x, test_y = dataset.get_test_data()
    test_summary = sess.run(all_summaries,
                            feed_dict={x: test_x,
                                       y_target: test_y,
                                       dropout_switch: 0})
    test_writer.add_summary(test_summary, batches_processed)

    for _ in trange(epochs, desc="{}_{}".format(optim_name, oargs).strip("_")):
        for bx, by in dataset.train_batches():
            batches_processed += 1
            # TODO workaround
            if pretrain_step is not None:
                sess.run(pretrain_step, feed_dict={x: bx, dropout_switch: 1})
            train_summary, _ = sess.run([train_summaries, train_step],
                                        feed_dict={x: bx,
                                                   y_target: by,
                                                   dropout_switch: 1})
            train_writer.add_summary(train_summary, batches_processed)
        # TODO change it to minibatches
        test_x, test_y = dataset.get_test_data()
        test_summary = sess.run(all_summaries,
                                feed_dict={x: test_x,
                                           y_target: test_y,
                                           dropout_switch: 0})
        test_writer.add_summary(test_summary, batches_processed)

    train_writer.close()
    test_writer.close()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default=DEFAULT_TB_LOGDIR,
        help="Summaries log directory")

    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        metavar="YAML_FILE",
        type=str,
        default="configs/mnist_all.yml",
        help="config file in yaml")
    parser.add_argument(
        "--tag",
        metavar="tag",
        type=str,
        help="runtag for tensorboard",
        default=None)

    args = parser.parse_args()

    config = defaultdict(lambda: None, yaml.safe_load(open(args.config)))
    optimizers = []

    for optim_class, instances in config["optimizers"].items():
        if not isinstance(instances, list):
            optimizers.append((optim_class, {}))
        else:
            for optim_args in instances:
                if optim_args is None:
                    optim_args = {}
                optimizers.append((optim_class, optim_args))

    print("Optimizers:", len(optimizers))
    print("Models:", len(config["models"]))
    print("Tests in total:", len(optimizers) * len(config["models"]))

    if "datasets" not in config:
        datasets = [config["dataset"]]
    else:
        datasets = config["datasets"]

    for dataset_name in datasets:
        dataset = eval(dataset_name)(**config)

        for model_class in config["models"]:
            print("Running optimizers for dataset: '{}', model: '{}'".format(dataset.get_name(), model_class))
            for optimizer_class, optimizer_args in sorted(optimizers, key=lambda x: x[0]):
                for _ in range(config["times"]):
                    try:
                        test(
                            args.logdir,
                            dataset,
                            model_class,
                            optimizer_class,
                            optimizer_args,
                            tag=args.tag,
                            **config)
                    except Exception as ex :
                        print("ERROR: {} crashed".format(optimizer_class))

