#!/usr/bin/env python3

import traceback
import argparse
import ruamel.yaml as yaml
from time import strftime
import tabulate
from collections import defaultdict
from models import *
from datasets import *
from short_names import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DEFAULT_TIMES = 1
DEFAULT_TB_LOGDIR = "tb_logs"
DEFAULT_LOGDIR = None
FLUSH_SECS = 2
TESTBATCH_SIZE = 100000
DEFAULT_EPOCHS = 30
DEFAULT_ONE_HOT = True


def _parse_list_dict(list_or_dict):
    if isinstance(list_or_dict, list):
        return [(k, {}) for k in list_or_dict]
    elif isinstance(list_or_dict, dict):
        return_list = []
        for key_class, instances in list_or_dict.items():
            if not isinstance(instances, list):
                return_list.append((key_class, {}))
            else:
                for args in instances:
                    if args is None:
                        args = {}
                    return_list.append((key_class, args))
        return return_list
    else:
        raise ValueError("no list/dict")


def test(
        tblogdir,
        logdir,
        dataset,
        model,
        model_args,
        optimizer_class,
        optimizer_args,
        epochs=DEFAULT_EPOCHS,
        one_hot=DEFAULT_ONE_HOT,
        train_histograms=False,
        tag=None,
        train_logs=True,
        no_tqdm=False,
        verbose=False,
        *args,
        **kwargs):
    # TODO add tag support
    if tag is not None:
        raise NotImplementedError()
    if logdir is not None:
        raise NotImplementedError()
        # tf.gfile.MakeDirs(logdir)
    tf.gfile.MakeDirs(tblogdir)
    tf.reset_default_graph()
    dropout_switch = tf.placeholder_with_default(1.0,
                                                 None,
                                                 name='dropout_switch')

    x = tf.placeholder(tf.float32, [None] + dataset.input_shape, name='x-input')
    y = eval(model)(
        x,
        dataset.outputs_num,
        dropout_switch=dropout_switch,
        **model_args)

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
    # TODO add model args to writer dirs

    # Summaries
    summaries_prefix = dataset.get_name()
    grad_hist_summaries = []
    var_hist_summaries = []
    for grad, var in grads_and_vars:
        g_summary = tf.summary.histogram('{}/{}/gradients/{}'.format(summaries_prefix, model, var.name), grad)
        v_summary = tf.summary.histogram('{}/{}/{}'.format(summaries_prefix, model, var.name), var)
        grad_hist_summaries.append(g_summary)
        var_hist_summaries.append(v_summary)
    acc_summary = tf.summary.scalar('{}/accuracy'.format(summaries_prefix), accuracy)
    ce_summary = tf.summary.scalar('{}/cross_entropy'.format(summaries_prefix), mean_cross_entropy)

    if train_histograms:
        train_summaries = tf.summary.merge_all()
    else:
        train_summaries = tf.summary.merge([acc_summary, ce_summary])
    test_summaries = tf.summary.merge([acc_summary, ce_summary] + var_hist_summaries)

    time = strftime("%m.%d_%H-%M-%S")
    optim_name = optimizer.get_name().lower()
    oargs = "_".join(k[0] + str(v) for k, v in sorted(optimizer_args.items()))
    prefix = "{}/{}/{}/{}/{}_{}".format(tblogdir, dataset.get_name(), model, time, optim_name, oargs)
    prefix = prefix.strip("_")
    if train_logs:
        train_writer = tf.summary.FileWriter(prefix + '/train', flush_secs=FLUSH_SECS)
    else:
        train_writer = None
    test_writer = tf.summary.FileWriter(prefix + '/test', flush_secs=FLUSH_SECS)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batches_processed = 0

    test_x, test_y = dataset.get_test_data()
    test_summary = sess.run(test_summaries,
                            feed_dict={x: test_x,
                                       y_target: test_y,
                                       dropout_switch: 0})
    test_writer.add_summary(test_summary, batches_processed)
    if no_tqdm:
        def trange(n, *args, **kwargs):
            for epoch in range(n):
                print("Epoch {}/{}".format(epoch + 1, n))
                yield epoch

    else:
        from tqdm import trange

    for _ in trange(epochs, desc="{}_{}".format(optim_name, oargs).strip("_")):
        for bx, by in dataset.train_batches():
            batches_processed += 1
            # TODO workaround
            if pretrain_step is not None:
                sess.run(pretrain_step, feed_dict={x: bx, dropout_switch: 1})
            if train_logs:
                train_summary, _ = sess.run([train_summaries, train_step],
                                            feed_dict={x: bx,
                                                       y_target: by,
                                                       dropout_switch: 1})
                train_writer.add_summary(train_summary, batches_processed)
            else:
                sess.run(train_step,
                         feed_dict={x: bx,
                                    y_target: by,
                                    dropout_switch: 1})
        # TODO change it to minibatches
        test_x, test_y = dataset.get_test_data()
        test_summary = sess.run(all_summaries,
                                feed_dict={x: test_x,
                                           y_target: test_y,
                                           dropout_switch: 0})
        test_writer.add_summary(test_summary, batches_processed)

    if train_writer is not None:
        train_writer.close()
    test_writer.close()
    sess.close()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--tblogdir",
            "-tbl",
            type=str,
            default=DEFAULT_TB_LOGDIR,
            help="TB Summaries log directory")
        parser.add_argument(
            "--logdir",
            "-l",
            type=str,
            default=DEFAULT_LOGDIR,
            help="Ordinary logs directory (NOT IMPLEMENTED")
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=False,
            help="Well... guess."
        )
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
            help="runtag for tensorboard (not implemented)",
            default=None)
        parser.add_argument(
            "--show-datasets",
            "-s",
            action="store_true",
            help="shows some descriptions of loaded datasets and exits",
            default=False,
        )

        args = parser.parse_args()
        if args.logdir is not None:
            raise NotImplementedError("logdir ...")

        if args.tag is not None:
            raise NotImplementedError("tag ...")

        config = defaultdict(lambda: None, yaml.safe_load(open(args.config)))
        optimizers = _parse_list_dict(config["optimizers"])
        models = _parse_list_dict(config["models"])

        if args.verbose:
            print("Optimizers:")
            for optim, oargs in optimizers:
                oargs = ", ".join(["{}:{}".format(k, v) for k, v in oargs.items()])
                print("\t{}  {}".format(optim, oargs))

            print("Models:")
            for model, margs in models:
                margs = ", ".join(["{}:{}".format(k, v) for k, v in margs.items()])
                print("\t{}  {}".format(model, margs))

        print("# Optimizers:", len(optimizers))
        print("# Models:", len(config["models"]))

        print("Tests in total:", len(optimizers) * len(config["models"]))

        if "datasets" not in config:
            if "dataset" in config:
                datasets = [config["dataset"]]
            else:
                raise ValueError("No dataset(s) specified.")
        else:
            datasets = config["datasets"]

        if args.show_datasets:
            header = "Name", "size", "features", "classes", "scale ", "spread"
            lines = []
            for dataset_name in datasets:
                dataset = eval(dataset_name)(**config)
                line = [dataset_name,
                        dataset.size,
                        np.prod(dataset.input_shape),
                        max(dataset.outputs_num, 2),
                        dataset.feature_scale,
                        dataset.feature_spread
                        ]
                lines.append(line)
            print(tabulate.tabulate(lines, header, floatfmt=".2E"))
            exit(0)
        for dataset_name in datasets:
            dataset = eval(dataset_name)(**config)
            for model_class, model_args in models:
                print("Running optimizers for dataset: '{}', model: '{}'".format(dataset.get_name(), model_class))
                for optimizer_class, optimizer_args in sorted(optimizers, key=lambda x: x[0]):
                    for _ in range(config["times"]):
                        try:
                            test(
                                args.tblogdir,
                                dataset,
                                model_class,
                                model_args,
                                optimizer_class,
                                optimizer_args,
                                tag=args.tag,
                                verbose=args.verbose,
                                **config)
                        except Exception as ex:
                            print("=============== EXCETPION ===============")
                            print("=============== {} ===============".format(optimizer_class))
                            print(ex)
                            traceback.print_exc(file=sys.stdout)
                            print("==========================================")
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Aborting ...")
