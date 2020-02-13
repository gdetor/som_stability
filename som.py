#!/usr/bin/env python
# This script implements the numerical experiments from [1]. It simulates a
# neural field self-organizing map.
# Copyright (C) <2020>  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# [1] "Stability analysis of a neural field self-organizing map",
#      G. Is. Detorakis, A. Chaillet, N.P. Rougier, 2020.
import sys
import time
import numpy as np
from functools import wraps
import matplotlib.pylab as plt
from numpy.fft import rfft2, irfft2

from plot_functions import plot_weights


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1-t0)))
        return result
    return function_timer


# FFT optimization function.
def best_fft_shape(shape):
    # fftpack (not sure of the base)
    base = [13, 11, 7, 5, 3, 2]

    # fftw
    # base = [13,11,7,5,3,2]

    def factorize(n):
        if n == 0:
            raise(RuntimeError, "Length n must be positive integer")
        elif n == 1:
            return [1, ]
        factors = []
        for b in base:
            while n % b == 0:
                n /= b
                factors.append(b)
        if n == 1:
            return factors
        return []

    def is_optimal(n):
        factors = factorize(n)
        # fftpack
        return len(factors) > 0

    shape = np.atleast_1d(np.array(shape))
    for i in range(shape.size):
        while not is_optimal(shape[i]):
            shape[i] += 1
    return shape.astype(int)


# A Gaussian-like function.
def gaussian(shape, width=(1, 1), center=(0, 0)):
    grid = []
    for size in shape:
        grid.append(slice(0, size))
    C = np.mgrid[tuple(grid)]
    R = np.zeros(shape)
    for i, size in enumerate(shape):
        if shape[i] > 1:
            R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return np.exp(-R/2)


@fn_timer
def simulation(store=False):
    # Parameters
    # --------------------------------------------
    Rn = 2      # Receptors count (Rn x Rn)
    n = 16      # Neural field size (n x n)
    p = 2*n+1

    T = 10.0       # Nu of Euler's time discretization 100.0
    ms = 0.001
    dt = 25.0*ms    # Timestep 35.0/T
    lrate = 0.01    # Learning rate 0.01
    alpha = 0.10    # Time constant
    tau = 1.00      # Synapse temporal decay
    epochs = 7000   # Number of training epochs

    Ke = 1.3        # Strength of lateral excitatory weights
    sigma_e = 0.1   # Extent of lateral excitatory weights
    Ki = 0.7        # Strength of lateral inhibitory weights
    sigma_i = 1.0   # Extent of lateral excitatory weights

    # Neural field setup
    # --------------------------------------------
    U = np.random.uniform(0.00, 0.01, (n, n))
    W = np.random.uniform(-1.0, 1.0, (n*n, Rn))

    # FFT implementation
    # --------------------------------------------
    We = Ke * gaussian((p, p), (sigma_e, sigma_e))
    Wi = Ki * gaussian((p, p), (sigma_i, sigma_i))

    U_shape, We_shape = np.array(U.shape), np.array(We.shape)
    shape = np.array(best_fft_shape(U_shape + We_shape//2))

    We_fft = rfft2(We[::-1, ::-1], shape)
    Wi_fft = rfft2(Wi[::-1, ::-1], shape)

    i0 = We.shape[0]//2
    i1 = i0+U_shape[0]
    j0 = We.shape[1]//2
    j1 = j0+U_shape[1]

    # Samples generation
    # --------------------------------------------
    n_samples = epochs
    samples = np.random.uniform(-1.0, 1.0, (n_samples, 2))

    # Actual training
    # --------------------------------------------
    w_hist = []
    w_hist.append(W.copy())
    for e in range(epochs):
        # Pick a random sample
        stimulus = samples[e]

        # Computes field input accordingly
        D = ((np.abs(W - stimulus)).sum(axis=-1))/float(Rn)
        Inp = (1.0 - D.reshape(n, n)) * alpha

        # Field simulation until convergence
        for i in range(int(T / dt)):
            Z = rfft2(np.maximum(U, 0.0), shape)
            Le = irfft2(Z * We_fft, shape).real[i0:i1, j0:j1]
            Li = irfft2(Z * Wi_fft, shape).real[i0:i1, j0:j1]
            U += (-U + (Le - Li) + Inp) * dt / tau
            W -= lrate * (Le.ravel() * (W - stimulus).T).T * dt
            w_hist.append(W.copy())

        # Field's activity reset
        U = np.random.uniform(0.00, 0.01, (n, n))

        # Store weights and show statistics
        if store is True:
            np.save("./data/weights_"+str(e), W.flatten())
        if e % 500 == 0:
            print(e)

    w_hist = np.array(w_hist)
    np.save("./data/w_hist_unstable", w_hist)
    return samples, W


if __name__ == '__main__':
    np.random.seed(int(sys.argv[1]))
    # np.random.seed(37)
    s, w = simulation()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_weights(s, w.reshape(16, 16, 2), ax)
    plt.show()
