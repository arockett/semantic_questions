
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

DTYPE = tf.float32

def lazy_property(function):
    attribute = "_" + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function())
        return getattr(self, attribute)
    return wrapper


def weight_and_bias(in_size, out_size, name = ""):
    """Create a weight and bias pair of variables with an optional name prefix."""
    weight_name = "{}_w".format(name)
    bias_name = "{}_b".format(name)

    weight = tf.truncated_normal_initializer([in_size, out_size], stddev=0.01, dtype=DTYPE)
    weight = tf.get_variable(weight_name, initializer=weight)

    bias = tf.constant_initializer(0.1, shape=[out_size], dtype=DTYPE)
    bias = tf.get_variable(bias_name, initializer=bias)

    return weight, bias

