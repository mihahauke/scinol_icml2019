import tensorflow as tf
from tensorflow.python.framework import ops
from ._scinol import _BaseOptimizer

SMALL_NUMBER = 1e-10
DEFAULT_UNPUTS_SUFFIX = "input"


class PreScinolOptimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self,
                 alpha=1.125,
                 epsilon=1.0,
                 epsilon_scaled=False,
                 s0=0,
                 name="PreScinol",
                 use_locking=False):
        super(PreScinolOptimizer, self).__init__(use_locking, name)
        self.alpha = alpha
        self.epsilon = epsilon
        self.s0 = s0
        if epsilon_scaled not in [False,"d","dt"]:
            raise ValueError("Improper epsilon scaled: {}".format(epsilon_scaled))
        self.epsilon_scaled = epsilon_scaled

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self._get_or_make_slot(v, v, "initial_value", self._name)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        if x.shape == []:
            x2 = x ** 2
        else:
            x = tf.expand_dims(x, len(x.shape))
            x2 = tf.reduce_mean(x ** 2, 0)
            x2 = tf.broadcast_to(x2, var.get_shape())

        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")

        broadcasted_x2 = tf.broadcast_to(x2, s2.shape)
        s2 = tf.assign_add(s2, broadcasted_x2)
        t = self.t +1
        if self.epsilon_scaled == "d":
            d = float(var.get_shape().as_list()[0])
            epsilon = self.epsilon / d
        elif self.epsilon_scaled == "dt":
            d = float(var.get_shape().as_list()[0])
            epsilon = self.epsilon / (tf.to_float(t)*d)
        else:
            epsilon = self.epsilon
        new_var = epsilon * h / (self.alpha * s2) * tf.exp((h ** 2 + x2) / (2 * self.alpha * s2))

        # equivalent new_var[s2==0] = 0
        new_var = tf.where(tf.not_equal(s2, 0), new_var, tf.zeros_like(new_var))
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        h = self.get_slot(var, "grads_sum")
        new_h = tf.assign_add(h, -grad)
        return new_h


class PreScinol2Optimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self,
                 alpha=1.5,
                 epsilon=1,
                 s0=0,
                 name="PreScinol2",
                 use_locking=False):
        super(PreScinol2Optimizer, self).__init__(use_locking, name)
        self.alpha = alpha
        self.epsilon = epsilon
        self.s0 = s0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "eta", self.epsilon)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        if x.shape == []:
            x2 = x ** 2
        else:
            x = tf.expand_dims(x, len(x.shape))
            x2 = tf.reduce_mean(x ** 2, 0)
            x2 = tf.broadcast_to(x2, var.get_shape())

        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")

        gamma = eta / self.alpha * tf.exp(-(h ** 2 * x2) / (s2 * (s2 + x2) * 2 * self.alpha))
        gamma = tf.where(tf.not_equal(s2, 0), gamma, eta / self.alpha)

        broadcasted_x2 = tf.broadcast_to(x2, s2.shape)
        new_s2 = tf.assign_add(s2, broadcasted_x2)

        new_var = gamma * h / new_s2
        new_var = tf.where(tf.not_equal(s2, 0), new_var, tf.zeros_like(new_var))
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        h = self.get_slot(var, "grads_sum")
        eta = self.get_slot(var, "eta")

        new_h = tf.assign_add(h, -grad)
        new_eta = tf.assign_add(eta, -grad * var)

        return tf.group(new_h, new_eta)


class PreScinolDLOptimizer(_BaseOptimizer):
    def __init__(self,
                 alpha=1.5,
                 epsilon=1.0,
                 s0=SMALL_NUMBER,
                 name="PreScinolDL",
                 use_locking=False):
        super(PreScinolDLOptimizer, self).__init__(use_locking, name)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.s0 = float(s0)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self._get_or_make_slot(v, v, "initial_value", self._name)

    def _apply_dense(self, grad, var):
        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")
        var0 = self.get_slot(var, "initial_value")

        new_h = tf.assign_add(h, -grad)
        new_s2 = tf.assign_add(s2, grad ** 2)

        new_var = var0 + self.epsilon * new_h / (self.alpha * new_s2) * tf.exp(new_h ** 2 / (2 * self.alpha * new_s2))
        new_var = tf.assign(var, new_var)

        return tf.group(new_var, new_h, new_s2)


class PreScinol2DLOptimizer(_BaseOptimizer):
    def __init__(self,
                 alpha=1.5,
                 epsilon=1.0,
                 s0=SMALL_NUMBER,
                 name="PreScinolDL",
                 use_locking=False):
        super(PreScinol2DLOptimizer, self).__init__(use_locking, name)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.s0 = float(s0)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "eta", self.epsilon)
                self._get_or_make_slot(v, v, "initial_value", self._name)

    def _apply_dense(self, grad, var):
        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")
        var0 = self.get_slot(var, "initial_value")

        new_h = tf.assign_add(h, -grad)
        new_s2 = tf.assign_add(s2, grad ** 2)
        new_eta = tf.maximum(self.epsilon, eta - (var - var0) * grad)

        new_var = var0 + new_eta * new_h / (self.alpha * new_s2)
        new_var = tf.assign(var, new_var)

        return tf.group(new_var, new_h, new_s2, new_eta)
