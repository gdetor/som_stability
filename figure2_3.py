# This script generates the Figures No 2 and No 3 in [1].
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


from plot_functions import plot_panels


if __name__ == '__main__':
    w_hist = np.load('./data/w_hist_stable.npy')
    distortion = np.load('./data/distortion_stable.npy')
    plot_panels(w_hist, distortion)
    plt.savefig('Figure02.pdf', axis='tight')

    # w_hist = np.load('./data/w_hist_unstable.npy')
    # distortion = np.load('./data/distortion_unstable.npy')
    # plot_panels(w_hist, distortion)
    # plt.savefig('Figure03.pdf', axis='tight')
    plt.show()
