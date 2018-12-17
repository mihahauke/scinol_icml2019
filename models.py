from tensorflow.contrib.layers import fully_connected

import tensorflow as tf


class _Model(object):
    def __init__(self, name=None):
        self.name = name

    def _model(self, inputs, outputs_num, dropout_switch):
        raise NotImplementedError()

    def __call__(self, inputs, outputs_num, dropout_switch):
        return self._model(inputs, outputs_num, dropout_switch)


# Wrappers that set inputs needed for scinol algos
def _fc(inputs, scope, *args, **kwargs):
    inputs = tf.identity(inputs, name="{}/weights/input".format(scope))
    return fully_connected(inputs, scope=scope, *args, **kwargs)


def _conv(inputs, scope, *args, **kwargs):
    inputs = tf.identity(inputs, name="{}/weights/input".format(scope))

    return tf.layers.conv2d(
        inputs,
        name=scope,
        *args,
        **kwargs
    )



class LR(_Model):
    def __init__(self,
                 name=None,
                 init0=False):
        if init0:
            self.initializer = tf.initializers.zeros
            if name is None:
                name = "lr0"
        else:
            # Glorot by default
            self.initializer = None
            if name is None:
                name = "lr"

        super(LR, self).__init__(
            name)

    def _model(self, inputs, outputs_num, dropout_switch):
        inputs = tf.layers.flatten(inputs)

        return _fc(inputs,
                   scope="fc_lr",
                   num_outputs=outputs_num,
                   activation_fn=None,
                   biases_initializer=tf.initializers.zeros,
                   weights_initializer=tf.initializers.zeros)


class NN(_Model):
    def __init__(self,
                 layers,
                 name=None,
                 dropout=0.9,
                 batch_norm=False,
                 activation_fn=tf.nn.relu):
        if name is None:
            name = "nn_{}_d{:0.2f}_b{}".format("x".join([str(l) for l in layers]), dropout, int(batch_norm))
        super(NN, self).__init__(name)
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.layers = layers

    def _model(self, inputs, outputs_num, dropout_switch):
        inputs = tf.layers.flatten(inputs)
        keep_prob = 1 - (1 - self.dropout) * dropout_switch
        for i, num_units in enumerate(self.layers):
            inputs = _fc(inputs,
                         num_outputs=num_units,
                         scope="fc{}".format(i),
                         activation_fn=self.activation_fn)
            if self.batch_norm:
                inputs = tf.layers.batch_normalization(inputs)
            if self.dropout > 0:
                inputs = tf.nn.dropout(inputs, keep_prob)

        return _fc(inputs, "fc_out", num_outputs=outputs_num, activation_fn=None)


class CharLSTM(_Model):
    def __init__(self,
                 layers=(128, 128),
                 rnn_class=tf.contrib.rnn.LSTMCell,
                 name=None,
                 dropout=0.0,
                 batch_norm=False,
                 activation_fn=tf.nn.relu):
        if name is None:
            name = "rnn_{}_d{:0.2f}_b{}".format("x".join([str(l) for l in layers]), dropout, int(batch_norm))
        super(CharLSTM, self).__init__(name)
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.layers = layers
        self.rnn_class = rnn_class

    def _model(self, inputs, outputs_num, dropout_switch):
        keep_prob = 1 - (1 - self.dropout) * dropout_switch
        for i, num_units in enumerate(self.layers):
            inputs = tf.identity(inputs, name="rnn_cell{}/kernel/input".format(i))
            cell = self.rnn_class(num_units, name="rnn_cell{}".format(i))

            if self.batch_norm:
                inputs = tf.layers.batch_normalization(inputs)
            if self.dropout > 0:
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                     input_keep_prob=keep_prob,
                                                     output_keep_prob=keep_prob)
            inputs, states = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=inputs,
                                               time_major=True,
                                               # initial_state = cell_wrapped.zero_state(batch_size=None,
                                               #                                         dtype=tf.float32),
                                               dtype=tf.float32,
                                               scope="rnn_unwind")

        fc = _fc(inputs, "fc_out", num_outputs=outputs_num, activation_fn=None)
        return fc


class CNN(_Model):
    def __init__(self,
                 filters_nums,
                 kernel_sizes,
                 strides,
                 fc_layers,
                 pooling=None,
                 dropout=0.9,
                 batch_norm=False,
                 name=None,
                 activation_fn=tf.nn.relu):
        if name is None:
            desc = "_".join(kernel_sizes) + "_" + "x".join(fc_layers)
            name = "cnn_{}_d{:0.2f}_b{}".format(desc, dropout, int(batch_norm))

        super(CNN, self).__init__(
            name)
        if len(kernel_sizes) != len(strides) or len(kernel_sizes) != len(filters_nums):
            raise ValueError()
        if len(fc_layers) == 0:
            raise ValueError()
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.filters_nums = filters_nums
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.fc_layers = fc_layers
        self.pooling = pooling,

    def _model(self, inputs, outputs_num, dropout_switch):
        keep_prob = 1 - (1 - self.dropout) * dropout_switch
        for i, [filters_num, kernel_size, stride] in enumerate(zip(self.filters_nums, self.kernel_sizes, self.strides)):
            inputs = _conv(
                inputs,
                strides=stride,
                filters=filters_num,
                kernel_size=kernel_size,
                scope="conv_{}".format(i),
                data_format='channels_last',
                padding="VALID",
                activation=self.activation_fn
            )
            if self.dropout > 0:
                inputs = tf.nn.dropout(inputs, keep_prob)
            if self.batch_norm:
                inputs = tf.layers.batch_normalization(inputs)
            if self.pooling is not None:
                inputs = tf.nn.max_pool(inputs, ksize=[1, self.pooling, self.pooling, 1], strides=[1, 2, 2, 1],
                                        padding='SAME')

        inputs = tf.layers.flatten(inputs)

        for i, num_units in enumerate(self.fc_layers):
            inputs = _fc(inputs,
                         num_outputs=num_units,
                         scope="fc{}".format(i),
                         activation_fn=self.activation_fn)
            if self.batch_norm:
                inputs = tf.layers.batch_normalization(inputs)
            if self.dropout > 0:
                inputs = tf.nn.dropout(inputs, keep_prob)
        return _fc(inputs, "fc_out", num_outputs=outputs_num, activation_fn=None)


# def cocob_cnn(inputs, outputs_num, dropout_switch=0.0):
#     return cnn(
#         inputs,
#         outputs_num,
#         dropout_switch,
#         dropout=0.5,
#         filters=[32, 64],
#         kernels=[5, 5],
#         strides=[1, 1],
#         pooling=2,
#         fc_layers=[1024]
#     )
#
#
# def cnn_simple(inputs, outputs_num, dropout_switch):
#     return cnn(
#         inputs,
#         outputs_num,
#         dropout_switch,
#         filters=[64, 32],
#         kernels=[4, 3],
#         strides=[2, 1],
#         fc_layers=[500]
#     )
#
#
# def cnn_simple_elu(inputs, outputs_num, dropout_switch):
#     return cnn(
#         inputs,
#         outputs_num,
#         dropout_switch,
#         filters=[64, 32],
#         kernels=[4, 3],
#         strides=[2, 1],
#         fc_layers=[500],
#         activation_fn=tf.nn.elu
#     )


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
