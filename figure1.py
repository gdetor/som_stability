# This script generates Figure No 1 from [1].
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

from plot_functions import plot_weights


if __name__ == '__main__':
    samples = np.random.uniform(-1, 1, (2000, 2))
    epochs = ['0', '1000', '2000', '3000', '4000', '6999']
    cases = ['A', 'B', 'C', 'D', 'E']

    fig = plt.figure(figsize=(18, 3))
    for i in range(5):
        weights = np.load("./data/weights_"+epochs[i]+".npy").reshape(16,
                                                                      16,
                                                                      2)
        ax = fig.add_subplot(1, 5, i+1)
        plot_weights(samples, weights, ax)
        ax.text(-1, 1.2, cases[i],
                ha='left',
                va='top',
                weight='bold',
                fontsize=18)
    plt.savefig("Figure01.pdf", axis='tight')
    plt.show()
