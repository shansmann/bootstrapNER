import tensorflow as tf
import keras.backend as K
from keras.legacy import interfaces
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras.optimizers import Optimizer
from keras.initializers import Initializer


class ProbabilityConstraint(Constraint):
    """This class implements the Probability Constraint

    Matrix is first normalized to [0, 1] (Min/Max Norm.)
    after which a unit norm on row level is enforced
    """

    def __init__(self):
        pass

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        weights = w / (K.epsilon() + K.sqrt(K.sum(K.square(w),
                                                  axis=1,
                                                  keepdims=True)))
        #non_neg = (w - K.min(w)) / (K.max(w) - K.min(w))
        #weights = non_neg / (K.epsilon() + K.sqrt(K.sum(K.square(non_neg),
        #                                          axis=1,
        #                                          keepdims=True)))

        #weights = K.tf.nn.softmax(w, axis=1)
        #noisy_weights = w
        #noisy_weights += K.random_normal_variable(shape=w.shape, mean=0, scale=.1, seed=42)
        #weights = tf.exp(noisy_weights) / tf.reduce_sum(tf.exp(noisy_weights), 1)

        return weights

class TraceRegularizer(Regularizer):
    """This class implements the Trace regularizer.

    # Arguments:
        lambda: float >= 0
    """

    def __init__(self, lamb):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, weight_matrix):
        return self.lamb * tf.trace(weight_matrix)

    def get_config(self):
        return {'lambda': float(self.lamb)}


class TraceL2Regularizer(Regularizer):
    """This class implements the Trace L2 regularizer.

    # Arguments:
        lambda: float >= 0
        l2: float >= 0
    """

    def __init__(self, lamb, l2):
        self.lamb = K.cast_to_floatx(lamb)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, weight_matrix):
        regularization = 0.
        regularization += self.l2 * K.sum(K.square(weight_matrix))
        regularization += self.lamb * tf.trace(weight_matrix)
        return regularization

    def get_config(self):
        return {'lambda': float(self.lamb),
                'l2': float(self.l2)}

# taken from https://erikbrorson.github.io/2018/04/30/Adam-with-learning-rate-multipliers/
class SGD_lr_mult(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, multipliers=None, debug_verbose=False, **kwargs):
        super(SGD_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            # different learning rates
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None

            if multiplier:
                new_lr_t = lr * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], K.get_value(new_lr_t)))
            else:
                new_lr_t = lr
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))

            v = self.momentum * m - new_lr_t * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr_t * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'multipliers': self.multipliers}
        base_config = super(SGD_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NumpyInitializer(Initializer):
    """Initializer base class: all initializers inherit from this class.
    """
    def __init__(self, values):
        self.values = K.variable(values)

    def __call__(self, shape, dtype=None):
        return self.values