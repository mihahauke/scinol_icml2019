import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class AdalbertOptimizer(optimizer.Optimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, alpha=0.5, s0=0, use_locking=False, name="Adalbert"):
        super(AdalbertOptimizer, self).__init__(use_locking, name)

        self._alpha = float(alpha)
        self._s0 = s0

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
                self._create_const_init_slot(v, "squared_grads_sum", self._s0)
                self._get_or_make_slot(v, v, "initial_var", self._name)

    def _apply_dense(self, grad, var):
        var_0 = self.get_slot(var, "initial_var")
        grads_sum = self.get_slot(var, "grads_sum")
        squared_grads_sum = self.get_slot(var, "squared_grads_sum")

        new_grads_sum = tf.assign_add(grads_sum, grad)
        new_squared_grads_sum = tf.assign_add(squared_grads_sum, grad ** 2)
        new_squared_grads_sum_sqrt = (new_squared_grads_sum + 1) ** 0.5

        eta = tf.exp(self._alpha * tf.abs(new_grads_sum) / new_squared_grads_sum_sqrt) - 1
        # just for purpose of having a named op TODO is it necessary?
        eta = tf.identity(eta, name="eta")

        new_var = -self._alpha / new_squared_grads_sum_sqrt * tf.sign(new_grads_sum) * eta

        return tf.assign(var, var_0 + new_var)


class AdalbertOptimizer2(AdalbertOptimizer):
    def __init__(self, alpha=0.5, s0=0, use_locking=False, name="Adalbert_Momentum_0"):
        super(AdalbertOptimizer2, self).__init__(alpha=alpha, s0=s0, use_locking=use_locking, name=name)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._create_const_init_slot(v, "grads_sum", 0)
                self._create_const_init_slot(v, "squared_grads_sum", self._s0)
                self._get_or_make_slot(v, v, "initial_var", self._name)
                self._get_or_make_slot(v, v, "averaged_var", self._name)

    def _apply_dense(self, grad, var):
        grads_sum = self.get_slot(var, "grads_sum")
        squared_grads_sum = self.get_slot(var, "squared_grads_sum")
        var_0 = self.get_slot(var, "initial_var")
        var_avg = self.get_slot(var, "averaged_var")

        new_grads_sum = tf.assign_add(grads_sum, grad)
        grad_squared = grad ** 2
        new_squared_grads_sum = tf.assign_add(squared_grads_sum, grad_squared)

        eta = tf.exp(self._alpha * tf.abs(new_grads_sum) / (new_squared_grads_sum + 1) ** 0.5) - 1
        eta = tf.identity(eta, name="eta")

        new_var = var_0 - self._alpha / tf.sqrt(new_squared_grads_sum + 1) * tf.sign(grads_sum) * eta
        new_var = tf.assign(var, new_var + var_avg)
        new_var_avg = (var_avg * (squared_grads_sum + 1) + grad_squared * new_var) / (new_squared_grads_sum + 1)
        return new_var, new_var_avg


class AdalbertOptimizer3(AdalbertOptimizer):
    def __init__(self, alpha=0.5, s0=0, maxgrad0=1e-8, use_locking=False, name="Adalbert_Modified"):
        super(AdalbertOptimizer3, self).__init__(alpha=alpha, s0=s0, use_locking=use_locking, name=name)
        self._maxgrad0 = maxgrad0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._zeros_slot(v, "grads_sum", self._name)
                self._zeros_slot(v, "squared_grads_sum", self._name)
                self._create_const_init_slot(v, "maxgrad", self._maxgrad0)
                self._get_or_make_slot(v, v, "initial_var", self._name)

    def _apply_dense(self, grad, var):
        grads_sum = self.get_slot(var, "grads_sum")
        squared_grads_sum = self.get_slot(var, "squared_grads_sum")
        var_0 = self.get_slot(var, "initial_var")
        maxgrad = self.get_slot(var, "maxgrad")

        grad_squared = grad ** 2
        new_maxgrad = tf.assign(maxgrad, tf.maximum(maxgrad, grad_squared))
        new_grads_sum = tf.assign_add(grads_sum, grad)
        new_squared_grads_sum = tf.assign_add(squared_grads_sum, grad_squared)
        new_squared_grads_sum_sqrt = tf.sqrt(new_squared_grads_sum + new_maxgrad)

        eta = (tf.exp(self._alpha * tf.abs(new_grads_sum) / new_squared_grads_sum_sqrt) - 1)
        eta = tf.identity(eta, name="eta")

        new_var = - self._alpha / new_squared_grads_sum_sqrt * tf.sign(new_grads_sum) * eta
        new_var = tf.assign(var, var_0 + new_var)

        return new_var


class NewAlgorithmOptimizer(AdalbertOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, s0=1, eta0=1, use_locking=False, name="newalgorithm"):
        super(NewAlgorithmOptimizer, self).__init__(s0=s0, use_locking=use_locking, name=name)

        self._eta0 = eta0

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._get_or_make_slot(v, v, "initial_var", self._name)
                self._create_const_init_slot(v, "squared_grads_sum", self._s0)
                self._create_const_init_slot(v, "grads_sum", 0)
                self._create_const_init_slot(v, "reward", self._eta0)

    def _apply_dense(self, grad, var):
        grads_sum = self.get_slot(var, "grads_sum")
        squared_grads_sum = self.get_slot(var, "squared_grads_sum")
        reward = self.get_slot(var, "reward")
        var_0 = self.get_slot(var, "initial_var")

        new_reward = tf.assign(reward, tf.maximum(reward - grad * var, 0.5 * reward))

        new_grads_sum = tf.assign_add(grads_sum, grad)
        new_squared_grads_sum = tf.assign_add(squared_grads_sum, grad ** 2)
        new_squared_grads_sum_sqrt = tf.sqrt(new_squared_grads_sum)

        denominator = 2 * tf.maximum(tf.maximum(new_squared_grads_sum, 1),
                                     new_squared_grads_sum_sqrt * tf.abs(new_grads_sum))
        new_var = -new_reward * new_grads_sum / denominator
        new_var = tf.assign(var, var_0 + new_var)

        return new_var

class NewAlgorithmLikeCocob(AdalbertOptimizer):
    """Optimizer that implements the <NAME_HERE> algorithm.

    See this [paper](TODO)
    """

    def __init__(self, alpha=50, use_locking=False, name="CocobLike"):
        super(NewAlgorithmLikeCocob, self).__init__(use_locking, name=name)

        self._alpha = float(alpha)

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                self._create_const_init_slot(v, "grads_sum", 0)
                self._create_const_init_slot(v, "squared_grads_sum", 0)
                self._create_const_init_slot(v, "reward", 0)
                self._create_const_init_slot(v, "maxgrad", 1e-8)
                self._get_or_make_slot(v, v, "initial_var", self._name)

    def _apply_dense(self, grad, var):
        var_0 = self.get_slot(var, "initial_var")
        grads_sum = self.get_slot(var, "grads_sum")
        squared_grads_sum = self.get_slot(var, "squared_grads_sum")
        reward = self.get_slot(var, "reward")
        maxgrad = self.get_slot(var, "maxgrad")

        grad_squared = grad ** 2
        new_reward = tf.assign(reward, tf.maximum(reward - grad * (var - var_0), 0))
        new_grads_sum = tf.assign_add(grads_sum, grad)
        new_squared_grads_sum = tf.assign_add(squared_grads_sum, grad_squared)
        new_maxgrad = tf.assign(maxgrad, tf.maximum(maxgrad, grad_squared))
        new_squared_grads_sum_sqrt = tf.sqrt(new_squared_grads_sum + new_maxgrad)

        denominator = 2 * tf.maximum(new_maxgrad + new_squared_grads_sum,
                                     tf.maximum(new_squared_grads_sum_sqrt * tf.abs(new_grads_sum),
                                                self._alpha * new_maxgrad))
        new_var = -(new_reward + tf.sqrt(new_maxgrad)) * new_grads_sum / denominator
        return tf.assign(var, var_0 + new_var)
