#! /usr/bin/env python3
"""Sarcos regression demo with TensorBoard."""
import logging

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import aboleth as ab
from aboleth.likelihoods import Normal
from aboleth.datasets import fetch_gpml_sarcos_data


# Set up a python logger so we can see the output of MonitoredTrainingSession
logger = logging.getLogger()
logger.setLevel(logging.INFO)

NSAMPLES = 10  # Number of random samples to get from an Aboleth net
NFEATURES = 2000  # Number of random features/bases to use in the approximation
VARIANCE = 10.0  # Initial estimate of the observation variance

# Random Fourier Features, this is setting up an anisotropic length scale, or
# one length scale per dimension
LENSCALE = ab.pos(tf.Variable(10 * np.ones((21, 1), dtype=np.float32)))
KERNEL = ab.RBF(LENSCALE)

# Variational Fourier Features -- length-scale setting here is the "prior"
# LENSCALE = 10.
# KERNEL = ab.RBFVariational(lenscale=LENSCALE, lenscale_posterior=LENSCALE)

# Build the approximate GP
net = ab.stack(
    ab.InputLayer(name='X', n_samples=NSAMPLES),
    ab.RandomFourier(n_features=NFEATURES, kernel=KERNEL),
    ab.DenseVariational(output_dim=1, full=True)
)

# Learning and prediction settings
BATCH_SIZE = 100  # number of observations per mini batch
NEPOCHS = 50  # Number of times to iterate though the dataset
NPREDICTSAMPLES = 10  # results in NSAMPLES * NPREDICTSAMPLES samples

CONFIG = tf.ConfigProto(device_count={'GPU': 1})  # Use GPU ?


def main():
    """Run the demo."""
    data = fetch_gpml_sarcos_data()
    Xr = data.train.data.astype(np.float32)
    Yr = data.train.targets.astype(np.float32)[:, np.newaxis]
    Xs = data.test.data.astype(np.float32)
    Ys = data.test.targets.astype(np.float32)[:, np.newaxis]
    N, D = Xr.shape

    print("Iterations: {}".format(int(round(N * NEPOCHS / BATCH_SIZE))))

    # Scale and centre the data, as per the original experiment
    ss = StandardScaler()
    Xr = ss.fit_transform(Xr)
    Xs = ss.transform(Xs)
    ym = Yr.mean()
    Yr -= ym
    Ys -= ym

    # Data
    with tf.name_scope("Input"):
        Xb, Yb = batch_training(Xr, Yr, n_epochs=NEPOCHS,
                                batch_size=BATCH_SIZE)
        X_ = tf.placeholder_with_default(Xb, shape=(None, D))
        Y_ = tf.placeholder_with_default(Yb, shape=(None, 1))

    with tf.name_scope("Likelihood"):
        var = ab.pos(tf.Variable(VARIANCE))
        lkhood = Normal(variance=var)

    with tf.name_scope("Deepnet"):
        Phi, kl = net(X=X_)
        loss = ab.elbo(Phi, Y_, N, kl, lkhood)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging
    log = tf.train.LoggingTensorHook(
        {'step': global_step, 'loss': loss},
        every_n_iter=1000
    )

    with tf.train.MonitoredTrainingSession(
            config=CONFIG,
            checkpoint_dir="./sarcos/",
            save_summaries_steps=None,
            save_checkpoint_secs=60,
            save_summaries_secs=20,
            hooks=[log]
    ) as sess:
        try:
            while not sess.should_stop():
                sess.run(train)
        except tf.errors.OutOfRangeError:
            print('Input queues have been exhausted!')
            pass

        # Prediction
        Ey = ab.predict_samples(Phi, feed_dict={X_: Xs, Y_: np.zeros_like(Ys)},
                                n_groups=NPREDICTSAMPLES, session=sess)
        sigma2 = var.eval(session=sess)

    # Score
    Eymean = Ey.mean(axis=0)
    Eyvar = Ey.var(axis=0) + sigma2  # add sigma2 for obervation noise
    r2 = r2_score(Ys.flatten(), Eymean)
    snlp = msll(Ys.flatten(), Eymean, Eyvar, Yr.flatten())

    print("------------")
    print("r-square: {:.4f}, smse: {:.4f}, msll: {:.4f}."
          .format(r2, 1 - r2, snlp))


def msll(Y_true, Y_pred, V_pred, Y_train):
    """Mean standardised log loss."""
    mt, st = Y_train.mean(), Y_train.std()
    ll = norm.logpdf(Y_true, loc=Y_pred, scale=np.sqrt(V_pred))
    rand_ll = norm.logpdf(Y_true, loc=mt, scale=st)
    msll = - (ll - rand_ll).mean()
    return msll


def batch_training(X, Y, batch_size, n_epochs):
    """Batch training queue."""
    X = tf.train.limit_epochs(X, n_epochs, name="X_lim")
    Y = tf.train.limit_epochs(Y, n_epochs, name="Y_lim")
    X_batch, Y_batch = tf.train.shuffle_batch([X, Y], batch_size, 100, 1,
                                              enqueue_many=True)
    return X_batch, Y_batch


if __name__ == "__main__":
    main()
