#!/usr/bin/env python3

from ._scinol import _PreApplyOptimizer, SMALL_NUMBER
import tensorflow as tf
from tensorflow.python.framework import ops


class NAGOptimizer(_PreApplyOptimizer):
    """Optimizer that implements the NAG algorithm.

    See this [paper](https://arxiv.org/abs/1305.6646)
    """

    def __init__(self, s0=SMALL_NUMBER, g0=SMALL_NUMBER, learning_rate=0.1, name="NAG", use_locking=False):
        super(NAGOptimizer, self).__init__(use_locking=use_locking, name=name)
        self.eta = learning_rate
        self.s0 = s0
        self.g0 = g0
        self.N = tf.Variable(self.s0, trainable=False)
        self.t = tf.train.get_or_create_global_step()

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "s", self.s0)
                self.create_const_init_slot(v, "G", self.g0)

    def _preapply_dense(self, var):
        x, x2 = self.inputs[var]
        s = self.get_slot(var, "s")
        new_s = tf.assign(s, tf.maximum(s, tf.abs(x)))
        new_var = tf.assign(var, var * (s / new_s))
        new_N = tf.assign_add(self.N, tf.reduce_sum(x2 / new_s ** 2))

        return tf.group(new_var, new_N)

    def _apply_dense(self, grad, var):
        s = self.get_slot(var, "s")
        G = self.get_slot(var, "G")
        N = self.N
        t = tf.to_float(tf.assign_add(self.t, 1))

        G = tf.assign_add(G, grad ** 2)

        new_var = tf.assign_add(var, -self.eta * (t / N) ** 0.5 * grad / (s * G ** 0.5))
        return new_var


class sNAGOptimizer(_PreApplyOptimizer):
    """Optimizer that implements the sNAG algorithm.
    See this [paper](https://arxiv.org/abs/1305.6646)
    """

    def __init__(self, name="sNAG", use_locking=False, **kwargs):
        super(sNAGOptimizer, self).__init__(use_locking=use_locking, name=name, **kwargs)

    def _apply_dense(self, grad, var):
        s = self.get_slot(var, "s")
        G = self.get_slot(var, "G")
        t = self.t
        N = self.N

        G = tf.assign_add(G, grad ** 2)

        new_var = tf.assign_add(var, -self.eta * (t / N) ** 0.5 * grad / ((s / t * G) ** 0.5))
        return new_var

    def _preapply_dense(self, var):
        x = self.inputs[var]
        s = self.get_slot(var, "s")

        new_s = tf.assign(s, x ** 2)
        new_var = tf.assign(var, var * s / new_s)
        new_N = tf.assign_add(self.N, tf.reduce_sum((x / new_s) ** 2))
        new_t = tf.assign_add(self.t, 1)

        return new_var, new_N, new_t
