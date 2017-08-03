
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .utils import lazy_property, DTYPE
from .query_model import QueryModel


class SiameseQuestionMatcher(object):

    def __init__(self, data1, data2, target, num_hidden=200, dropout=0.5):
        self.question1 = data1
        self.question2 = data2
        self.target = target
        self.dropout = dropout
        self._keep_prob = 1 - dropout
        self._num_hidden = num_hidden
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        pass

    @lazy_property
    def cost(self):
        pass

    @lazy_property
    def optimize(self):
        pass

    @lazy_property
    def error(self):
        pass

