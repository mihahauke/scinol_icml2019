#!/usr/bin/env python3

from ._scinol import _FeatureBasedOptimizer, SMALL_NUMBER
import tensorflow as tf
from tensorflow.python.framework import ops


class NAGOptimizer(_FeatureBasedOptimizer):
    """Optimizer that implements the NAG algorithm.

    See this TODO
    """

    def __init__(self, s0=SMALL_NUMBER, g0=SMALL_NUMBER, learning_rate=0.1, name="NAG", use_locking=False):
        super(NAGOptimizer, self).__init__(use_locking=use_locking, name=name)
        self.eta = learning_rate
        self.s0 = s0
        self.g0 = g0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "S2", self.s0)
                self.create_const_init_slot(v, "G", self.g0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)

    def _preapply_dense(self, var):
        x, _, max_x = self._process_inputs(var)
        S2 = self.get_slot(var, "S2")
        G = self.get_slot(var, "G")
        M = self.get_slot(var, "max")

        new_M = tf.assign(M, tf.maximum(M, max_x))
        d = 1
        new_var = tf.assign(var, -self.eta * G / (d ** 0.5 * S2 ** 0.5 * new_M ** 2))

        return new_var

    def _apply_dense(self, grad, var):
        S2 = self.get_slot(var, "S2")
        G = self.get_slot(var, "G")
        M = self.get_slot(var, "max")
        new_G = tf.assign_add(G, grad ** 2)
        new_S2 = tf.assign_add(S2, grad ** 2 / M ** 2)
        return new_S2, new_G

# class sNAGOptimizer(_BaseOptimizer):
#     """Optimizer that implements the sNAG algorithm.
#     See this [paper](https://arxiv.org/abs/1305.6646)
#     """
#
#     def __init__(self, name="sNAG", use_locking=False, **kwargs):
#         super(sNAGOptimizer, self).__init__(use_locking=use_locking, name=name, **kwargs)
#
#     def _apply_dense(self, grad, var):
#         s = self.get_slot(var, "s")
#         G = self.get_slot(var, "G")
#         t = self.t
#         N = self.N
#
#         G = tf.assign_add(G, grad ** 2)
#
#         new_var = tf.assign_add(var, -self.eta * (t / N) ** 0.5 * grad / ((s / t * G) ** 0.5))
#         return new_var
#
#     def _preapply_dense(self, var):
#         x = self.inputs[var]
#         if x.shape != []:
#             x = tf.expand_dims(x, len(x.shape))
#             x = tf.reduce_mean(x, 0)
#         s = self.get_slot(var, "s")
#
#         new_s = tf.assign(s, x ** 2)
#         new_var = tf.assign(var, var * s / new_s)
#         new_N = tf.assign_add(self.N, tf.reduce_sum((x / new_s) ** 2))
#         new_t = tf.assign_add(self.t, 1)
#
#         return new_var, new_N, new_t
