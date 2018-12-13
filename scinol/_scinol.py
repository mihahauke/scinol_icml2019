import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util import nest
from tensorflow.python.training.optimizer import Optimizer

SMALL_NUMBER = 1e-5
DEFAULT_UNPUTS_SUFFIX = "input"


class _BaseOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super(_BaseOptimizer, self).__init__(*args, **kwargs)
        self.inputs = None

    def create_const_init_slot(self, v, name, value=0):
        initializer = tf.initializers.constant(value, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

    def create_normal_init_slot(self, v, name, m=0, std=1):
        initializer = tf.initializers.random_normal(mean=m, stddev=std, dtype=v.dtype)

        self._get_or_make_slot_with_initializer(
            v, initializer, v.shape, v.dtype, name, self._name)

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

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        # TODO so many lines just to get var_list :/?
        if callable(loss):
            with backprop.GradientTape() as tape:
                if var_list is not None:
                    tape.watch(var_list)
                loss_value = loss()

                # Scale loss if using a "mean" loss reduction and multiple towers.
                # Have to be careful to call distribute_lib.get_loss_reduction()
                # *after* loss() is evaluated, so we know what loss reduction it uses.
                # TODO(josh11b): Test that we handle weight decay in a reasonable way.
                if (distribute_lib.get_loss_reduction() ==
                        variable_scope.VariableAggregation.MEAN):
                    num_towers = distribution_strategy_context.get_distribution_strategy(
                    ).num_towers
                    if num_towers > 1:
                        loss_value *= (1. / num_towers)

            if var_list is None:
                var_list = tape.watched_variables()
            # TODO(jhseu): Figure out why GradientTape's gradients don't require loss
            # to be executed.
            with ops.control_dependencies([loss_value]):
                grads = tape.gradient(loss_value, var_list, grad_loss)
            return list(zip(grads, var_list))

        # Non-callable/Tensor loss case
        if context.executing_eagerly():
            raise RuntimeError(
                "`loss` passed to Optimizer.compute_gradients should "
                "be a function when eager execution is enabled.")

        # Scale loss if using a "mean" loss reduction and multiple towers.
        if (distribute_lib.get_loss_reduction() ==
                variable_scope.VariableAggregation.MEAN):
            num_towers = distribution_strategy_context.get_distribution_strategy(
            ).num_towers
            if num_towers > 1:
                loss *= (1. / num_towers)

        if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                                  Optimizer.GATE_GRAPH]:
            raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                             "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                             gate_gradients)
        self._assert_valid_dtypes([loss])
        if grad_loss is not None:
            self._assert_valid_dtypes([grad_loss])
        if var_list is None:
            var_list = (
                    variables.trainable_variables() +
                    ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        else:
            var_list = nest.flatten(var_list)
        # pylint: disable=protected-access
        var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
        # pylint: enable=protected-access
        if not var_list:
            raise ValueError("No variables to optimize.")

        if self.inputs is None:
            self._retrieve_inputs(var_list)
        self._create_slots(var_list)

        update_ops = [self._preapply_dense(var) for var in var_list]
        with tf.control_dependencies(update_ops):
            return super(_BaseOptimizer, self).compute_gradients(loss, var_list,
                                                                 gate_gradients,
                                                                 aggregation_method,
                                                                 colocate_gradients_with_ops,
                                                                 grad_loss)


class ScinolOptimizer(_BaseOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, epsilon=1.0, s0=0, name="ScInOl", beta=None, use_locking=False):
        super(ScinolOptimizer, self).__init__(use_locking, name)
        self.epsilon = float(epsilon)
        self.s0 = s0
        self.t = tf.train.get_or_create_global_step()
        self.beta = beta

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self.create_const_init_slot(v, "grads_sum", 0)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self._get_or_make_slot(v, v, "initial_var", self._name)
                self.create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self.create_const_init_slot(v, "beta", 1)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        if x.shape == []:
            max_x = tf.abs(x)
            x2 = x ** 2
        else:
            x = tf.expand_dims(x, len(x.shape))
            x2 = tf.reduce_mean(x ** 2, 0)
            max_x = tf.reduce_max(tf.abs(x), 0)
            x2 = tf.broadcast_to(x2, var.get_shape())

        beta = self.get_slot(var, "beta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")
        var0 = self.get_slot(var, "initial_var")
        t = tf.to_float(tf.assign_add(self.t, 1))

        M = tf.assign(M, tf.maximum(M, max_x))
        if self.beta is not None:
            beta = tf.constant(1.0)
        else:
            beta = tf.assign(beta, tf.minimum(beta, self.epsilon * (S2 + M ** 2) / (x2 * (t + 1))))

        theta = G / (S2 + M ** 2) ** 0.5
        new_var = (beta * tf.sign(theta)) / (2 * (S2 + M ** 2) ** 0.5) * (tf.exp(tf.abs(theta) / 2) - 1)
        return tf.assign(var, var0 + new_var)

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
                self._get_or_make_slot(v, v, "initial_var", self._name)
                self.create_const_init_slot(v, "squared_grads_sum", self.s0)
                self.create_const_init_slot(v, "var_max", SMALL_NUMBER)
                self.create_const_init_slot(v, "eta", self.epsilon)

    def _preapply_dense(self, var):
        x = self.inputs[var]
        if x.shape == []:
            max_x = tf.abs(x)
        else:
            x = tf.expand_dims(x, len(x.shape))
            max_x = tf.reduce_max(tf.abs(x), 0)

        eta = self.get_slot(var, "eta")
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        M = self.get_slot(var, "var_max")
        var0 = self.get_slot(var, "initial_var")

        M = tf.assign(M, tf.maximum(M, max_x))

        theta = G / (S2 + M ** 2) ** 0.5

        new_var = tf.sign(theta) * tf.minimum(tf.abs(theta), 1.0) / (2 * (S2 + M ** 2) ** 0.5) * eta
        return tf.assign(var, var0 + new_var)

    def _apply_dense(self, grad, var):
        G = self.get_slot(var, "grads_sum")
        S2 = self.get_slot(var, "squared_grads_sum")
        eta = self.get_slot(var, "eta")
        var0 = self.get_slot(var, "initial_var")

        G = tf.assign_add(G, -grad)
        S2 = tf.assign_add(S2, (grad) ** 2)
        eta = tf.assign_add(eta, -grad * (var - var0))

        return tf.group(G, S2, eta)

