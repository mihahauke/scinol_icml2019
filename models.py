from tensorflow.contrib.layers import fully_connected

import tensorflow as tf


# TODO?
# class Model(object):
#     def __init__(self, name, ):
#         self._name = name


def nn(inputs,
       outputs_num,
       dropout_switch,
       layers,
       dropout=0.9,
       activation_fn=tf.nn.relu):
    inputs = tf.layers.flatten(inputs)
    keep_prob = 1 - (1 - dropout) * dropout_switch
    for num_units in layers:
        inputs = fully_connected(inputs, num_units, activation_fn=activation_fn)

        if dropout and dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob)

    return fully_connected(inputs, outputs_num, activation_fn=None)


def cnn(inputs,
        outputs_num,
        dropout_switch,
        filters,
        kernels,
        strides,
        fc_layers,
        pooling=None,
        dropout=0.9,
        activation_fn=tf.nn.relu):
    keep_prob = 1 - (1 - dropout) * dropout_switch

    if len(kernels) != len(strides) or len(kernels) != len(filters):
        raise ValueError()
    if len(fc_layers) == 0:
        raise ValueError()

    for filters_num, kernel_size, stride in zip(filters, kernels, strides):
        inputs = tf.layers.conv2d(
            inputs,
            filters_num,
            kernel_size,
            stride,
            data_format='channels_last',
            padding="VALID",
            activation=activation_fn
        )
        if dropout and dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob)
        if pooling is not None:
            inputs = tf.nn.max_pool(inputs, ksize=[1, pooling, pooling, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')

    inputs = tf.layers.flatten(inputs)
    for num_units in fc_layers:
        inputs = fully_connected(inputs, num_units, activation_fn=activation_fn)
        if dropout and dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob)

    return fully_connected(inputs, outputs_num, activation_fn=None)


def lr(inputs, outputs_num, *args, **kwargs):
    inputs = tf.layers.flatten(inputs)
    return fully_connected(inputs, outputs_num, activation_fn=None)


def lr0(inputs, outputs_num, *args, **kwargs):
    inputs = tf.layers.flatten(inputs)
    return fully_connected(inputs,
                           outputs_num,
                           activation_fn=None,
                           biases_initializer=tf.initializers.zeros,
                           weights_initializer=tf.initializers.zeros)


def cocob_cnn(inputs, outputs_num, dropout_switch):
    return cnn(
        inputs,
        outputs_num,
        dropout_switch,
        dropout=0.5,
        filters=[32, 64],
        kernels=[5, 5],
        strides=[1, 1],
        pooling=2,
        fc_layers=[1024]
    )


def cnn_simple(inputs, outputs_num, dropout_switch):
    return cnn(
        inputs,
        outputs_num,
        dropout_switch,
        filters=[64, 32],
        kernels=[4, 3],
        strides=[2, 1],
        fc_layers=[500]
    )


def cnn_simple_elu(inputs, outputs_num, dropout_switch):
    return cnn(
        inputs,
        outputs_num,
        dropout_switch,
        filters=[64, 32],
        kernels=[4, 3],
        strides=[2, 1],
        fc_layers=[500],
        activation_fn=tf.nn.elu
    )


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

# def tf_cifar_cnn(inputs, outputs_num, *args, **kwargs):
#     conv1 = tf.layers.conv2d(
#         inputs,
#         64,
#         5,
#         1,
#         padding="SAME",
#         activation=tf.nn.relu
#     )
#
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool1')
#     # norm1
#     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm1')
#
#     # conv2
#     conv2 = tf.layers.conv2d(
#         inputs,
#         64,
#         5,
#         1,
#         padding="SAME",
#         activation=tf.nn.relu
#     )
#
#     # norm2
#     norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm2')
#     # pool2
#     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#     # local3
#     with tf.variable_scope('local3') as scope:
#         # Move everything into depth so we can perform a single matrix multiply.
#         # reshape = tf.reshape(pool2, [200, 2304])
#         reshape = tf.reshape(pool2, [tf.shape(inputs).get_shape().as_list()[0], -1])
#         dim = reshape.get_shape()[1].value
#         weights = _variable_with_weight_decay('weights', shape=[dim, 384],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#     # local4
#     with tf.variable_scope('local4') as scope:
#         weights = _variable_with_weight_decay('weights', shape=[384, 192],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = _variable_with_weight_decay('weights', [192, outputs_num],
#                                               stddev=1 / 192.0, wd=None)
#         biases = _variable_on_cpu('biases', [outputs_num],
#                                   tf.constant_initializer(0.0))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#
#     return softmax_linear
