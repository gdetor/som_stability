# This script provides all the necessary plot functions for generating figures
# 1, 2 and 3 in [1].
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

from som_dxdy import som_regularity

np.random.seed(560)     # 137 stable, 560 unstable


def plot_weights(samples, weights, ax):
    # Draw samples
    ax.scatter(samples[:, 0], samples[:, 1], s=3.0,
               color='k', alpha=0.3, zorder=1)

    # Draw network
    x, y = weights[..., 0], weights[..., 1]
    if len(weights.shape) > 2:
        for i in range(weights.shape[0]):
            ax.plot(x[i, :], y[i, :], 'k', alpha=0.95, lw=1.5, zorder=2)
        for i in range(weights.shape[1]):
            ax.plot(x[:, i], y[:, i], 'k', alpha=0.95, lw=1.5, zorder=2)
    else:
        ax.plot(x, y, 'k', alpha=0.85, lw=1.5, zorder=2)
    ax.scatter(x, y, s=50, c='w', edgecolors='k', zorder=3)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_panels(w_hist, distortion):
    w = w_hist[-1].flatten()
    n, k = 16, 2

    fig = plt.figure(figsize=(13, 13))
    fig.subplots_adjust(wspace=.3, hspace=.2)
    ax = fig.add_subplot(221)
    data = w.reshape(n, n, k)
    samples = np.random.uniform(-1, 1, (5000, 2))
    plot_weights(samples, data, ax)
    ax.text(-1, 1.1, 'A',
            ha='left',
            va='top',
            fontsize=18,
            weight='bold')

    ax = fig.add_subplot(222)
    dX, dY, SOMLine = som_regularity(w.reshape(16, 16, 2))
    ax.scatter(dY, dX, s=1, c='k', marker='s', label='Rate-distortion')
    x_ext = np.linspace(0, dY.max(), 100)
    p = np.polyfit(SOMLine[:, 0], SOMLine[:, 1], deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax.plot(x_ext, y_ext, 'r', lw=1.5)
    ax.set_xlim([0, dY.max()+0.15])
    ax.set_ylim([0, dX.max()+0.15])
    ax.set_xlabel(r'$\delta{\bf y}$', fontsize=15)
    ax.set_ylabel(r'$\delta{\bf x}$', fontsize=15)
    ax.set_xticks([0, 10, 20])
    ticks = ax.get_xticks()
    ax.set_xticklabels(ticks, fontsize=15, weight='bold')
    ticks = ax.get_yticks()
    ax.set_yticklabels(ticks, fontsize=15, weight='bold')
    ax.text(0, 2.26, 'B',
            ha='left',
            va='top',
            fontsize=18,
            weight='bold')

    ax = fig.add_subplot(223)
    ax.plot(distortion, 'k', lw=2, label='Rate-distortion')
    ax.set_xlim([0, 7000])
    # ax.set_ylim([0, 0.05])
    ax.set_ylim([0, 0.2])
    ax.set_xticks([0, 3500, 7000])
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=15, weight='bold')
    ticks = ax.get_yticks()
    ticks = [np.round(i, 2) for i in ticks]
    ax.set_yticklabels(ticks, fontsize=15, weight='bold')
    ax.set_xlabel('Epochs', fontsize=15, weight='bold')
    ax.set_ylabel('Distortion', fontsize=15, weight='bold')
    # ax.text(0, 0.0525, 'C',
    ax.text(0, 0.21, 'C',
            ha='left',
            va='top',
            fontsize=18,
            weight='bold')

    End = 7000 * 400
    colors = ['k', 'orange', 'm']
    Neurons = [(10, 10), (4, 9), (14, 3), (10, 10), (4, 9), (14, 3)]
    ax = fig.add_subplot(224)
    ii = 0
    for i in range(3):
        ax.plot(w_hist[:, np.prod(Neurons[i]), 0], color=colors[i], lw=2,
                label=r'${\bf r^*}$'+'='+str(Neurons[i]))
        ax.set_ylim([-2, 2])
        ax.set_xlim([0, End])

        ticks = ax.get_yticks()
        ticks = [np.round(i, 2) for i in ticks]
        ax.set_yticklabels(ticks, fontsize=15, weight='bold')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(r'${\bf w_f(r^*, t)}$', rotation=90, fontsize=15,
                      weight='bold')
        ticks = ax.get_xticks().astype('i')
        ticks = [int(i / 400) for i in ticks]
        ax.set_xticklabels(ticks, fontsize=15, weight='bold')
        ax.set_xlabel('Epochs', fontsize=15, weight='bold')
        ax.legend()
        ii += 1
    # ax.text(0, 1.1, 'D',
    ax.text(0, 2.19, 'D',
            ha='left',
            va='top',
            fontsize=18,
            weight='bold')
