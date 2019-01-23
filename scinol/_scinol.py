import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.training.optimizer import Optimizer

SMALL_NUMBER = 1e-15
DEFAULT_UNPUTS_SUFFIX = "input"


class _BaseOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super(_BaseOptimizer, self).__init__(*args, **kwargs)
        self.inputs = None
        self.t = tf.train.get_or_create_global_step()

    def create_const_init_slot(self, v, name, value=0):
        initializer = tf.initializers.constant(value, dtype=v.dtype)

        return self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

    def create_normal_init_slot(self, v, name, m=0, std=1):
        initializer = tf.initializers.random_normal(mean=m, stddev=std, dtype=v.dtype)

        return self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)


class _FeatureBasedOptimizer(_BaseOptimizer):
    def __init__(self,
                 use_locking=False,
                 name=None,
                 epsilon=1.0,
                 epsilon_scaled=False,
                 s0=0
                 ):
        super(_FeatureBasedOptimizer, self).__init__(use_locking=use_locking, name=name)
        self.epsilon = float(epsilon)
        self.epsilon_scaled = epsilon_scaled
        self.s0 = s0

    def setup_epsilon_slot(self, var, name):
        if not self.epsilon_scaled:
            return self.create_const_init_slot(var, name, 1.0)
        if len(var.shape) == 1:
            value = (1 / var.get_shape().as_list()[0]) ** 0.5
            return self.create_const_init_slot(var, name, value)
        else:
            initializer = tf.initializers.glorot_normal(dtype=var.dtype)

            return self._get_or_make_slot_with_initializer(
                var, initializer, var.shape, var.dtype, name, self._name)

    def _process_inputs(self, var):
        x = self.inputs[var]
        if x.shape == []:
            max_x = tf.abs(x)
            x2 = x ** 2
        else:
            x = tf.expand_dims(x, len(x.shape))
            x2 = tf.reduce_mean(x ** 2, 0)
            max_x = tf.reduce_max(tf.abs(x), 0)
            x2 = tf.broadcast_to(x2, var.get_shape())
        return x, x2, max_x

    def _retrieve_inputs(self, var_list):

        self.inputs = {}
        operations = {op.name: op for op in tf.get_default_graph().get_operations()}
        for var in var_list:
            op_name = var.op.name + "/{}".format("input")
            inputs = operations.get(op_name, None)
            if inputs is None:
                inputs = tf.constant(1.0, name=op_name)
            # TODO does it work in more general cases?
            if isinstance(inputs, tf.Operation):
                inputs = inputs.outputs[0]
            self.inputs[var] = inputs

    # def compute_gradients(self, loss, var_list=None,
    #                       gate_gradients=Optimizer.GATE_OP,
    #                       aggregation_method=None,
    #                       colocate_gradients_with_ops=False,
    #                       grad_loss=None):
    #
    #     # TODO perhaps retrieve lost copied code from here some day
    #     if var_list is None:
    #         var_list = (
    #                 variables.trainable_variables() +
    #                 ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    #     else:
    #         var_list = nest.flatten(var_list)
    #
    #     # pylint: disable=protected-access
    #     var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
    #     # pylint: enable=protected-access
    #     if not var_list:
    #         raise ValueError("No variables to optimize.")
    #
    #     if self.inputs is None:
    #         self._retrieve_inputs(var_list)
    #     self._create_slots(var_list)
    #     preapply_ops = [self._preapply_dense(var) for var in var_list]
    #
    #     t_op = tf.assign_add(self.t, 1)
    #     preapply_ops.append(t_op)
    #
    #     with tf.control_dependencies(preapply_ops):
    #         return super(_FeatureBasedOptimizer, self).compute_gradients(loss, var_list,
    #                                                                      gate_gradients,
    #                                                                      aggregation_method,
    #                                                                      colocate_gradients_with_ops,
    #                                                                      grad_loss)
    @property
    def preapply_ops(self):
        var_list = (
                variables.trainable_variables() +
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)

        if self.inputs is None:
            self._retrieve_inputs(var_list)
        self._create_slots(var_list)

        t_op = tf.assign_add(self.t, 1)
        with tf.control_dependencies([t_op]):
            preapply_ops = [self._preapply_dense(var) for var in var_list]

        return preapply_ops

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        unzipped_grads_and_vars = []
        for g, v in grads_and_vars:
            unzipped_grads_and_vars += [g, v]
        with tf.control_dependencies(unzipped_grads_and_vars):
            return super(_FeatureBasedOptimizer, self).apply_gradients(grads_and_vars, global_step, name)


class ScinolOptimizer(_FeatureBasedOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self,
                 name="ScInOl",
                 beta=None,
                 *args, **kwargs):
        super(ScinolOptimizer, self).__init__(name=name, *args, **kwargs)
        self.beta = beta

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self._get_or_make_slot(v, v, "initial_value", self._name)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "beta")
                self.setup_epsilon_slot(v, "epsilon")

    def _preapply_dense(self, var):
        _, x2, max_x = self._process_inputs(var)

        beta = self.get_slot(var, "beta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "max")
        var0 = self.get_slot(var, "initial_value")
        epsilon = self.get_slot(var, "epsilon")
        t = tf.to_float(self.t)

        new_M = tf.assign(M, tf.maximum(M, max_x))
        if self.beta is not None:
            beta = tf.constant(float(self.beta))
        else:
            beta = tf.assign(beta, tf.minimum(beta, (S2 + new_M ** 2) / (x2 * t)))

        theta = G / (S2 + new_M ** 2) ** 0.5
        new_var = (beta * tf.sign(theta)) / (2 * (S2 + new_M ** 2) ** 0.5) * (tf.exp(tf.abs(theta) / 2) - 1)
        return tf.assign(var, new_var)

    def _apply_dense(self, grad, var):
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")

        new_G = tf.assign_add(G, -grad)
        new_S2 = tf.assign_add(S2, (grad) ** 2)

        return new_G, new_S2


class Scinol2Optimizer(_FeatureBasedOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self,
                 name="ScInOl2",
                 *args, **kwargs
                 ):
        super(Scinol2Optimizer, self).__init__(name=name, *args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self._get_or_make_slot(v, v, "initial_value", self._name)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "eta")

    def _preapply_dense(self, var):
        x, _, max_x = self._process_inputs(var)
        eta = self.get_slot(var, "eta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "max")
        var0 = self.get_slot(var, "initial_value")

        new_M = tf.assign(M, tf.maximum(M, max_x))

        theta = G / (S2 + new_M ** 2) ** 0.5

        var_delta = tf.sign(theta) * tf.minimum(tf.abs(theta), 1.0) / (2 * (S2 + new_M ** 2) ** 0.5) * eta
        return tf.assign(var, var0 + var_delta)

    def _apply_dense(self, grad, var):
        x, _, max_x = self._process_inputs(var)
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")
        var0 = self.get_slot(var, "initial_value")

        new_G = tf.assign_add(G, -grad)
        new_S2 = tf.assign_add(S2, (grad) ** 2)
        new_eta = tf.assign_add(eta, -grad * (var - var0))

        # import sys
        # if "weights" in var.name:
        #     n = "W: "
        # else:
        #     n = "B:"
        # print_op = tf.print(
        #     n, "\t",
        #     tf.reduce_sum(var ** 2),
        #     "g:", tf.reduce_sum(grad),
        #     "x:", tf.reduce_sum(x),
        #     "G:", tf.reduce_sum(new_G),
        #     "S2:", tf.reduce_sum(new_S2),
        #     "eta:", tf.reduce_sum(new_eta),
        #     summarize=-1, output_stream=sys.stdout)
        # with ops.control_dependencies([print_op]):
        return tf.group(new_G, new_S2, new_eta)


class ScinolAOptimizer(ScinolOptimizer):
    """Inicjalizacja zgodnie z dokumentem new_alg.tex, tzn. S_0 np. rzędu 100 i potem początkowy skumulowany gradient G_i ~ N(0, S_0/d), gdzie S_0/d jest *wariancją*, a d = (n_in + n_out) / 2. Wartość początkową eta ustawiamy na 1.
    Hacky: this assumes that weights are initialized with glorot so it initializes G_i with s_0*V0 rather than ~ N(0, S_0/d) (basically the same result)
        """

    def __init__(self, s0=100, name="ScInOlA", *args, **kwargs):
        super(ScinolAOptimizer, self).__init__(s0=s0, name=name, *args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._get_or_make_slot(v, (self.s0) ** 0.5 * v, "grads_sum", self._name)
                self.create_const_init_slot(v, "initial_value", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "beta")
                self.setup_epsilon_slot(v, "epsilon")


class Scinol2AOptimizer(Scinol2Optimizer):
    """Inicjalizacja zgodnie z dokumentem new_alg.tex, tzn. S_0 np. rzędu 100 i potem początkowy skumulowany gradient G_i ~ N(0, S_0/d), gdzie S_0/d jest *wariancją*, a d = (n_in + n_out) / 2. Wartość początkową eta ustawiamy na 1.

    Hacky: this assumes that weights are initialized with glorot so it initializes G_i with s_0*V0 rather than ~ N(0, S_0/d) (basically the same result)
        """

    def __init__(self, s0=100, name="ScInOl2A", *args, **kwargs):
        super(Scinol2AOptimizer, self).__init__(s0=s0, name=name, *args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._get_or_make_slot(v, (self.s0) ** 0.5 * v, "grads_sum", self._name)
                self.create_const_init_slot(v, "initial_value", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "eta")


class ScinolBOptimizer(ScinolOptimizer):
    """Inicjalizacja podobnie jak w new_alg.tex, ale teraz S_0 = 1, G_i ~ N(0, 1),
a cała skala siedzi w zmiennej początkowej epsilon. Tzn. trzeba dobrać epsilon = sqrt(2/(n_in + n_out)).


    Hacky: this assumes that weights are initialized with glorot so it initializes eta with V0 rather than ~ N(0, 1/d) (basically the same result)
        """

    def __init__(self, s0=1, name="ScInOlB", *args, **kwargs):
        super(ScinolBOptimizer, self).__init__(s0=s0, epsilon_scaled=True, name=name, *args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_normal_init_slot(v, "grads_sum")
                self.create_const_init_slot(v, "initial_value", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "beta")
                self.setup_epsilon_slot(v, "epsilon")


class Scinol2BOptimizer(Scinol2Optimizer):
    """Inicjalizacja podobnie jak w new_alg.tex, ale teraz S_0 = 1, G_i ~ N(0, 1), a cała skala siedzi w zmiennej początkowej epsilon. Tzn. trzeba dobrać epsilon = sqrt(2/(n_in + n_out)).
    Hacky: this assumes that weights are initialized with glorot so it initializes eta with V0 rather than ~ N(0, 1/d) (basically the same result)

        """

    def __init__(self, s0=1, name="ScInOl2B", *args, **kwargs):
        super(Scinol2Optimizer, self).__init__(s0=s0, name=name, epsilon_scaled=True, *args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_normal_init_slot(v, "grads_sum")
                self.create_const_init_slot(v, "initial_value", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", SMALL_NUMBER)
                self.setup_epsilon_slot(v, "eta")


class Scinol2DLOptimizer(_BaseOptimizer):
    def __init__(self,
                 epsilon=1.0,
                 s0=0,
                 use_locking=False,
                 name="ScInOL2DL",
                 max_start=SMALL_NUMBER,
                 epsilon_scaled=False, ):
        super(Scinol2DLOptimizer, self).__init__(use_locking=use_locking, name=name)
        self.epsilon = float(epsilon)
        self.s0 = s0
        self.max_start = max_start
        self.epsilon_scaled = epsilon_scaled

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self._get_or_make_slot(v, v, "initial_value", self._name)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "max", self.max_start)

                if not self.epsilon_scaled:
                    self.create_const_init_slot(v, "eta", self.epsilon)
                else:
                    if len(v.shape) == 1:
                        value = (1 / v.get_shape().as_list()[0]) ** 0.5
                        self.create_const_init_slot(v, "eta", value)
                    else:
                        initializer = tf.initializers.glorot_normal(dtype=v.dtype)
                        self._get_or_make_slot_with_initializer(
                            v, initializer, v.shape, v.dtype, "eta", self._name)

    def _apply_dense(self, grad, var):
        eta = self.get_slot(var, "eta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "max")
        var0 = self.get_slot(var, "initial_value")

        M = tf.assign(M, tf.maximum(M, tf.abs(grad)))

        theta = G / (S2 + M ** 2) ** 0.5
        var_delta = tf.sign(theta) * tf.minimum(tf.abs(theta), 1.0) / (2 * (S2 + M ** 2) ** 0.5) * eta

        new_G = tf.assign_add(G, -grad)
        new_S2 = tf.assign_add(S2, (grad) ** 2)
        new_eta = tf.maximum(0.5 * eta, eta - grad * var_delta)
        new_eta = tf.assign(eta, new_eta)

        new_var = tf.assign(var, var0 + var_delta)

        return tf.group(new_G, new_S2, new_eta, new_var)
