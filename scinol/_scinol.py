import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

SMALL_NUMBER = 1e-6


class ScInOLOptimizer(optimizer.Optimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, epsilon=1.0, s0=0, name="ScInOl", use_locking=False, **kwargs):
        super(ScInOLOptimizer, self).__init__(use_locking, name)
        self.eps = float(epsilon)
        self.s0 = s0
        # TODO remove workaround:
        self.inputs = {}
        self.t = tf.Variable(0.0, trainable=False)

    def _create_const_init_slot(self, v, name, value=0):
        initializer = tf.initializers.constant(value, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

    def _create_normal_init_slot(self, v, name, m=0, std=1):
        initializer = tf.initializers.random_normal(mean=m, stddev=std, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._create_const_init_slot(v, "grads_sum", 0)
                self._create_const_init_slot(v, "squared_grads_sum", self.s0)
                # self._get_or_make_slot(v, v, "initial_var", self._name)
                self._create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self._create_const_init_slot(v, "beta", 1)

    # TODO HUGE workaround
    def pre_minimize(self, raw_features, var_list=None):
        self.raw_features = raw_features

        if var_list is None:
            var_list = tf.trainable_variables()

        self._create_slots(var_list)
        new_var_list = []
        for var in var_list:
            if "biases" in var.name:
                self.inputs[var] = tf.constant(1.0, shape=[1])
            else:
                # TODO avg???
                self.inputs[var] = tf.reshape(tf.reduce_mean(self.raw_features, axis=0), [-1, 1])
            new_var_list.append(self._preapply_dense(var))

        new_t = tf.assign_add(self.t, 1)
        return tf.group(new_var_list+ [new_t])

    def _preapply_dense(self, var):
        x = self.inputs[var]
        beta = self.get_slot(var, "beta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")

        M = tf.assign(M, tf.maximum(M, tf.abs(x)))
        beta = tf.assign(beta, tf.minimum(beta, self.eps * (S2 + M ** 2) / (x ** 2*(self.t+1))))

        theta = G / (S2 + M ** 2) ** 0.5
        new_var = (beta * tf.sign(theta)) / (2 * (S2 + M ** 2) ** 0.5) * (tf.exp(tf.abs(theta) / 2) - 1)
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        x = self.inputs[var]
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")

        G = tf.assign_add(G, -grad )
        S2 = tf.assign_add(S2, (grad ) ** 2)

        return G, S2


class ScInOL2Optimizer(ScInOLOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, name="ScInOl2", **kwargs):
        super(ScInOL2Optimizer, self).__init__(name=name, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._create_const_init_slot(v, "grads_sum", 0)
                self._create_const_init_slot(v, "squared_grads_sum", self.s0)
                # self._get_or_make_slot(v, v, "initial_var", self._name)
                self._create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self._create_const_init_slot(v, "eta", self.eps)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        eta = self.get_slot(var, "eta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")

        M = tf.assign(M, tf.maximum(M, tf.abs(x)))

        theta = G / (S2 + M ** 2) ** 0.5

        new_var = (tf.minimum(tf.abs(theta), 1) * tf.sign(theta)) / (2 * (S2 + M ** 2) ** 0.5) * eta
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        x = self.inputs[var]
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")

        G = tf.assign_add(G, -grad )
        S2 = tf.assign_add(S2, (grad) ** 2)
        eta = tf.assign_add(eta, -grad * var )

        return tf.group(G, S2, eta)


class NAGOptimizer(ScInOLOptimizer):
    """Optimizer that implements the sNAG algorithm.

    See this [paper](https://arxiv.org/abs/1305.6646)
    """

    def __init__(self, s0=SMALL_NUMBER, eta=0.1, name="NAG", use_locking=False, **kwargs):
        super(NAGOptimizer, self).__init__(s0=s0, use_locking=use_locking, name=name)
        self.eta = eta
        self.N = tf.Variable(self.s0, trainable=False)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._create_const_init_slot(v, "s", self.s0)
                self._create_const_init_slot(v, "G", self.s0)


    def _apply_dense(self, grad, var):
        s = self.get_slot(var, "s")
        G = self.get_slot(var, "G")
        t = self.t
        N = self.N

        G = tf.assign_add(G, grad ** 2)

        new_var = tf.assign_add(var, -self.eta * (t / N) ** 0.5 * grad / (s * G ** 0.5))
        return new_var

    def _preapply_dense(self, var):
        x = self.inputs[var]
        s = self.get_slot(var, "s")

        new_s = tf.assign_add(s, tf.maximum(s, tf.abs(x)))
        new_var = tf.assign(var, var * s / new_s)
        new_N = tf.assign_add(self.N, tf.reduce_sum((x / new_s) ** 2))
        new_t = tf.assign_add(self.t, 1)
        return new_var, new_N, new_t


class sNAGOptimizer(NAGOptimizer):
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

        new_s = tf.assign_add(s, x ** 2)
        new_var = tf.assign(var, var * s / new_s)
        new_N = tf.assign_add(self.N, tf.reduce_sum((x / new_s) ** 2))
        new_t = tf.assign_add(self.t, 1)

        return new_var, new_N, new_t
