# This script computes the distortion for the SOMs generated in [1] and
# based on equation (15) in [1].
# or it is violated.
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
import numpy as np
import matplotlib.pylab as plt

from numba import njit, prange

from sklearn.metrics import mean_squared_error


np.random.seed(37)     # 137


@njit(parallel=True)
def compute_rate_distortion(weights, samples):
    n_samples = samples.shape[0]
    distortion = 0
    for i in prange(n_samples):
        D = ((weights - samples[i])**2).sum(axis=-1)
        distortion += D.min()
    distortion /= n_samples
    return distortion


def mse(weights, epochs=6998, template=None):
    if template is None:
        w_final = weights[-1]
    else:
        w_final = template
    mse = []
    for i in range(epochs):
        mse.append(mean_squared_error(w_final, weights[i*400]))
    return np.array(mse)


if __name__ == '__main__':
    epochs = 7000

    if 0:
        # Compute the MSE based on the final SOM's weiights
        w_stable = np.load("./data/w_hist_stable.npy").reshape(-1, 16*16*2)
        w_unstable = np.load("./data/w_hist_unstable.npy").reshape(-1, 16*16*2)
        mse_stable = mse(w_stable)
        mse_unstable = mse(w_unstable, template=w_stable[-1])
        np.save("mse_stable", mse_stable)
        np.save("mse_unstable", mse_unstable)

    if 1:
        # Compute distortion
        w_stable = np.load("./data/w_hist_stable.npy").reshape(-1, 16*16, 2)
        # w_unstable = np.load("./data/w_hist_unstable.npy").reshape(-1,
        #                                                            16*16,
        #                                                            2)
        samples = np.random.uniform(-1, 1, (epochs, 2))
        distortion = []
        for i in range(epochs):
            distortion.append(compute_rate_distortion(w_stable[i*400],
                                                      samples))
        plt.plot(distortion)
        distortion = np.array(distortion)
        np.save('./data/distortion_stable', distortion)
        plt.show()
