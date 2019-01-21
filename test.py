#!/usr/bin/env python3

import traceback
import argparse
import ruamel.yaml as yaml
from time import strftime
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
REGRESSION_LOSSES = ("abs", "squared")
CLASSIFICATION_LOSSES = ("cross_entropy",)


# TODO parsing a list is not needed anymore . .. i think
def _parse_list_dict(list_or_dict):
    if isinstance(list_or_dict, list):
        return_list = []
        for obj in list_or_dict:
            if isinstance(obj, str):
                return_list.append((obj, {}))
            elif isinstance(obj, str):
                assert len(obj) == 1
                return_list.append((obj.keys()[0], {}))
            else:
                raise ValueError()
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


def _parse_name(name, args):
    s = name
    if len(args) > 0:
        s += str(args)

    return s


def test(
        dataset,
        model,
        model_args,
        optimizer_class,
        optimizer_args,
        tblogdir=DEFAULT_TB_LOGDIR,
        logdir=DEFAULT_LOGDIR,
        epochs=DEFAULT_EPOCHS,
        train_histograms=False,
        test_histograms=False,
        tag=None,
        train_logs=True,
        no_tqdm=False,
        embedding_size=None,
        loss=None,
        verbose=False,
        *args,
        **kwargs):
    # TODO add tag support
    if tag is not None:
        raise NotImplementedError()
    if logdir is not None:
        raise NotImplementedError()

    tf.gfile.MakeDirs(tblogdir)
    tf.reset_default_graph()
    dropout_switch = tf.placeholder_with_default(1.0,
                                                 None,
                                                 name='dropout_switch')

    if dataset.use_embeddings:
        x = tf.placeholder(tf.int32, [None] + dataset.input_shape, name='x-input')
        embeddings = tf.get_variable("embedding", [dataset.tokens_num, embedding_size],
                                     initializer=tf.random_normal_initializer, trainable=True)
        model_input = tf.nn.embedding_lookup(embeddings, x)
    else:
        x = tf.placeholder(tf.float32, [None] + dataset.input_shape, name='x-input')
        model_input = x
    model = eval(model)(**model_args)
    model_output = model(model_input, dataset.outputs_num, dropout_switch=dropout_switch)

    if dataset.task == CLASSIFICATION:
        if loss is None:
            loss = "cross_entropy"
        if loss == "cross_entropy":
            if dataset.sequential:
                # fold batchsize with sequence len
                seq_len = model_output.shape[1]
                logits_flat = tf.reshape(model_output, [-1, model_output.shape[2]])
                target = tf.placeholder(tf.int64, [None, seq_len], name='y-input')
                target_flat = tf.reshape(target, [-1])
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_flat, logits=logits_flat)
                correct_predictions = tf.equal(tf.argmax(logits_flat, 1), target_flat)
            else:
                # TODO check if changes work as expected
                if dataset.outputs_num == 1:
                    target = tf.placeholder(tf.float32, [None], name='y-input')
                    flat_y = tf.reshape(model_output, [-1])
                    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=flat_y)
                    correct_predictions = tf.equal(tf.cast(tf.greater(flat_y, 0), tf.float32), target)
                else:
                    target = tf.placeholder(tf.float32, [None, dataset.outputs_num], name='y-input')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=model_output)
                    correct_predictions = tf.equal(tf.argmax(model_output, 1), tf.argmax(target, 1))
            mean_cross_entropy = tf.reduce_mean(cross_entropy)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            loss_op = mean_cross_entropy
        elif loss not in CLASSIFICATION_LOSSES:
            raise ValueError("Loss for classification should be one of: {}, is: {}".format(CLASSIFICATION_LOSSES, loss))
        else:
            NotImplementedError()
    else:
        if dataset.sequential:
            raise NotImplementedError()
        else:
            target = tf.placeholder(tf.float32, [None, dataset.outputs_num], name='y-input')
            if loss is None:
                loss = "squared"
            if loss == "squared":
                loss_op = tf.reduce_mean((target - model_output) ** 2 / 2)
            elif loss == "abs":
                loss_op = tf.reduce_mean(tf.abs(target - model_output))
            elif loss not in REGRESSION_LOSSES:
                raise ValueError("Loss for regression should be one of: {}, is: {}".format(REGRESSION_LOSSES, loss))
            else:
                raise NotImplementedError()

    optimizer = eval(optimizer_class)(**optimizer_args)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    train_step = optimizer.apply_gradients(grads_and_vars)
    # TODO add model args to writer dirs

    # Summaries
    summaries_prefix = dataset.get_name()
    grad_hist_summaries = []
    var_hist_summaries = []
    for grad, var in grads_and_vars:
        g_summary = tf.summary.histogram('{}/{}/gradients/{}'.format(summaries_prefix, model.name, var.name),
                                         grad)
        v_summary = tf.summary.histogram('{}/{}/{}'.format(summaries_prefix, model.name, var.name), var)
        grad_hist_summaries.append(g_summary)
        var_hist_summaries.append(v_summary)

    loss_summary = tf.summary.scalar('{}/{}'.format(summaries_prefix, loss), loss_op)
    summaries = [loss_summary]
    if dataset.task == CLASSIFICATION:
        acc_summary = tf.summary.scalar('{}/accuracy'.format(summaries_prefix), accuracy)
        summaries.append(acc_summary)

    if train_histograms:
        train_summaries = tf.summary.merge_all()
    else:
        train_summaries = tf.summary.merge(summaries)
    if test_histograms:
        test_summaries = tf.summary.merge(summaries + var_hist_summaries)
    else:
        test_summaries = tf.summary.merge(summaries)

    time = strftime("%m.%d_%H-%M-%S")
    optim_name = optimizer.get_name().lower()
    oargs = "_".join(k[0] + str(v) for k, v in sorted(optimizer_args.items()))
    prefix = "{}/{}/{}/{}/{}_{}".format(tblogdir, dataset.get_name(), model.name, time, optim_name, oargs)
    prefix = prefix.strip("_")
    if train_logs:
        train_writer = tf.summary.FileWriter(prefix + '/train',
                                             graph=tf.get_default_graph(),
                                             flush_secs=FLUSH_SECS)
    else:
        train_writer = None
    test_writer = tf.summary.FileWriter(prefix + '/test',
                                        graph=tf.get_default_graph(),
                                        flush_secs=FLUSH_SECS)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batches_processed = 0
    test_x, test_y = dataset.get_test_data()
    pre_run_test_summary = sess.run(test_summaries,
                                    feed_dict={x: test_x,
                                               target: test_y,
                                               dropout_switch: 0})
    test_writer.add_summary(pre_run_test_summary, batches_processed)
    if no_tqdm:
        def trange(n, *_, **__):
            for epoch in range(n):
                print("Epoch {}/{}".format(epoch + 1, n))
                yield epoch

    else:
        from tqdm import trange

    for _ in trange(epochs, desc="{}_{}".format(optim_name, oargs).strip("_")):
        for bx, by in dataset.train_batches():
            batches_processed += 1
            if train_logs:
                train_summary, _ = sess.run([train_summaries, train_step],
                                            feed_dict={x: bx,
                                                       target: by,
                                                       dropout_switch: 1})
                train_writer.add_summary(train_summary, batches_processed)
            else:
                sess.run(train_step,
                         feed_dict={x: bx,
                                    target: by,
                                    dropout_switch: 1})
        # TODO change it to minibatches
        test_x, test_y = dataset.get_test_data()
        test_summary = sess.run(test_summaries,
                                feed_dict={x: test_x,
                                           target: test_y,
                                           dropout_switch: 0})
        test_writer.add_summary(test_summary, batches_processed)

    if train_writer is not None:
        train_writer.flush()
        train_writer.close()
    test_writer.flush()
    test_writer.close()
    sess.close()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
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

        if args.tag is not None:
            raise NotImplementedError("tag ...")

        try:
            config = defaultdict(lambda: None, yaml.safe_load(open(args.config)))
        except:
            print("Failed to load config!")
            traceback.print_exc(file=sys.stdout)
            print("Aborting!")
            exit(1)
        try:
            optimizers = _parse_list_dict(config["optimizers"])
        except ValueError:
            print("Failed to parse optimizers from config:")
            print(config["optimizers"])
            print("Aborting!")
            exit(1)
        try:
            models = _parse_list_dict(config["models"])
        except ValueError:
            print("Failed to parse models from config:")
            print(config["models"])
            print("Aborting!")
            exit(1)
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
                del config["dataset"]
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
            import tabulate

            print(tabulate.tabulate(lines, header, floatfmt=".2E"))
            exit(0)
        for dataset_name in datasets:
            dataset = eval(dataset_name)(**config)
            for model, model_args in models:
                print("Running optimizers for dataset: '{}', model: '{}'".format(dataset.get_name(),
                                                                                 _parse_name(model, model_args)))
                for optimizer_class, optimizer_args in sorted(optimizers, key=lambda x: x[0]):
                    for _ in range(config["times"]):
                        try:
                            test(
                                dataset=dataset,
                                model=model,
                                model_args=model_args,
                                optimizer_class=optimizer_class,
                                optimizer_args=optimizer_args,
                                tag=args.tag,
                                verbose=args.verbose,
                                **config)
                        except Exception as ex:
                            print("======================= EXCEPTION ===================================")
                            print("========================== {} =======================================".format(
                                optimizer_class))
                            print(ex)
                            traceback.print_exc(file=sys.stdout)
                            print("==========================================")
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt. Aborting ...")
