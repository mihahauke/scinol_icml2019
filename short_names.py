from scinol import *
import tensorflow as tf

from cocob import COCOBOptimizer, COCOBOptimizer0

nag = NAGOptimizer
scinol = ScinolOptimizer
scinol2 = Scinol2Optimizer
scinola = ScinolAOptimizer
scinol2a = Scinol2AOptimizer
scinolb = ScinolBOptimizer
scinol2b = Scinol2BOptimizer
prescinol = PreScinolOptimizer
prescinol2 = PreScinol2Optimizer
prescinoldl = PreScinolDLOptimizer
prescinol2dl = PreScinol2DLOptimizer
scinol2dl = Scinol2DLOptimizer
cocob = COCOBOptimizer
cocob0 = COCOBOptimizer0

rmsprop = tf.train.RMSPropOptimizer
adagrad = tf.train.AdagradOptimizer
adam = tf.train.AdamOptimizer
adadelta = tf.train.AdadeltaOptimizer


def sgd(learning_rate, use_locking=False, name="SGD", decay="sqrt",*args,**kwargs):
    if decay is not None:
        t = tf.train.get_or_create_global_step()
        t = tf.assign_add(t, 1)
        t = tf.to_float(t)
        if decay == "linear":
            learning_rate = learning_rate / t
        elif decay == "sqrt":
            learning_rate = learning_rate / tf.sqrt(t)
        else:
            raise ValueError("Unsupported decay for sgd: {}".format(decay))

    return tf.train.GradientDescentOptimizer(learning_rate, use_locking, name)
