"""Output likelihoods."""
import numpy as np
import tensorflow as tf

from aboleth.util import pos


def normal(variance):
    """Normal log-likelihood."""
    def loglike(x, f):
        ll = -0.5 * (tf.log(2 * variance * np.pi) + (x - f)**2 / variance)
        return ll
    return loglike


def bernoulli():
    """Bernoulli log-likelihood."""
    def loglike(x, f):
        ll = x * tf.log(pos(f)) + (1 - x) * tf.log(pos(1 - f))
        return ll
    return loglike


def binomial(n):
    """Binomial log-likelihood."""
    def loglike(x, f):
        bincoef = tf.lgamma(n + 1) - tf.lgamma(x + 1) - tf.lgamma(n - x + 1)
        ll = bincoef + x * tf.log(pos(f)) + (n - x) * tf.log(pos(1 - f))
        return ll
    return loglike