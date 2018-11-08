import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

SMALL_NUMBER = 1e-6


class _BaseOptimizer(optimizer.Optimizer):
    def __init__(self, use_locking, name):
        super(_BaseOptimizer, self).__init__(use_locking, name)
        self.t = tf.Variable(0.0, trainable=False)
        self.inputs = {}

    def create_const_init_slot(self, v, name, value=0):
        initializer = tf.initializers.constant(value, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

    def create_normal_init_slot(self, v, name, m=0, std=1):
        initializer = tf.initializers.random_normal(mean=m, stddev=std, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

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
        return tf.group(new_var_list + [new_t])


class ScinolOptimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, epsilon=1.0, s0=0, name="ScInOl", use_locking=False):
        super(ScinolOptimizer, self).__init__(use_locking, name)
        self.epsilon = float(epsilon)
        self.s0 = s0
        # TODO remove workaround:

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                # self._get_or_make_slot(v, v, "initial_var", self._name)
                self.create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self.create_const_init_slot(v, "beta", 1)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        beta = self.get_slot(var, "beta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")

        M = tf.assign(M, tf.maximum(M, tf.abs(x)))
        beta = tf.assign(beta, tf.minimum(beta, self.epsilon * (S2 + M ** 2) / (x ** 2 * (self.t + 1))))

        theta = G / (S2 + M ** 2) ** 0.5
        new_var = (beta * tf.sign(theta)) / (2 * (S2 + M ** 2) ** 0.5) * (tf.exp(tf.abs(theta) / 2) - 1)
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")

        G = tf.assign_add(G, -grad)
        S2 = tf.assign_add(S2, (grad) ** 2)

        return G, S2


class Scinol2Optimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, epsilon=1.0, s0=0, name="ScInOl2", use_locking=False):
        super(Scinol2Optimizer, self).__init__(use_locking, name)
        self.epsilon = float(epsilon)
        self.s0 = s0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self.create_const_init_slot(v, "eta", self.epsilon)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        eta = self.get_slot(var, "eta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")

        M = tf.assign(M, tf.maximum(M, tf.abs(x)))

        theta = G / (S2 + M ** 2) ** 0.5

        new_var = tf.sign(theta) * tf.minimum(tf.abs(theta), 1.0) / (2 * (S2 + M ** 2) ** 0.5) * eta
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")

        G = tf.assign_add(G, -grad)
        S2 = tf.assign_add(S2, (grad) ** 2)
        eta = tf.assign_add(eta, -grad * var)

        return tf.group(G, S2, eta)


class PreScinolOptimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self,
                 alpha=1.5,
                 epsilon=1,
                 s0=0,
                 name="PreScinol",
                 use_locking=False):
        super(PreScinolOptimizer, self).__init__(use_locking, name)
        self.alpha = alpha
        self.epsilon = epsilon
        self.s0 = s0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                # self._get_or_make_slot(v, v, "initial_var", self._name)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")

        broadcasted_x = tf.broadcast_to(x, s2.shape)
        s2 = tf.assign_add(s2, broadcasted_x ** 2)
        new_var = self.epsilon * h / (self.alpha * s2) * tf.exp((h ** 2 + x ** 2) / (2 * self.alpha * s2))

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
        h = self.get_slot(var, "grads_sum")
        s2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")


        gamma = eta /self.alpha  * tf.exp(-(h**2*x**2)/(s2*(s2+x**2)))
        gamma = tf.where(tf.not_equal(s2, 0), gamma, eta/self.alpha)

        broadcasted_x = tf.broadcast_to(x, s2.shape)
        new_s2 = tf.assign_add(s2, broadcasted_x ** 2)

        new_var = gamma*h/new_s2
        new_var = tf.where(tf.not_equal(s2, 0), new_var, tf.zeros_like(new_var))
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        h = self.get_slot(var, "grads_sum")
        eta = self.get_slot(var, "eta")

        new_h = tf.assign_add(h, -grad)
        new_eta = tf.assign_add(eta,-grad*var)

        return tf.group(new_h, new_eta)

class NAGOptimizer(_BaseOptimizer):
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
                self.create_const_init_slot(v, "s", self.s0)
                self.create_const_init_slot(v, "G", self.s0)

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
