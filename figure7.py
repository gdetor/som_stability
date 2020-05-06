# This script generates Figure No 6 from [1].
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
from som_dxdy import som_regularity

from sklearn.linear_model import LinearRegression


def euclidean_dist(x, y):
    return np.sqrt(((x - y)**2).sum())


if __name__ == '__main__':
    np.random.seed(135)
    epochs = 1000
    Ke = [0.47, 0.61, 0.74, 0.8, 0.9, 1.0, 2.0, 3.0]
    Ki = [0.2, 0.3, 0.4, 0.45, 0.53, 0.60, 1.40, 2.25]

    cases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    names = ['04702', '01603', '07404', '08045', '09053', '1006', '20135',
             '30225']
    KeKi = zip(Ke, Ki)
    labels = [r'($K_e=$'+str(i)+r', $K_i =$ '+str(j)+')' for i, j in KeKi]
    samples = np.random.uniform(0, 1, (epochs, 2))

    lreg = LinearRegression(fit_intercept=False)

    if 1:
        fig = plt.figure(figsize=(20, 7))
        fig.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.2)
        for i, name in enumerate(names):
            w = np.load("./results/w_hist_"+name+".npy")[-1]
            ax = fig.add_subplot(2, 8, i+1)
            plot_weights(samples, w.reshape(16, 16, 2), ax, axis=[0, 1],
                         size=10)
            ax.set_title(cases[i], fontsize=16, weight='bold')

        for i, name in enumerate(names):
            ax = fig.add_subplot(2, 8, 9+i)
            w = np.load("./results/w_hist_"+name+".npy")[-1]
            dX, dY, SOMLine = som_regularity(w.reshape(16, 16, 2))
            K = np.random.choice([i for i in range(len(dX))], size=5000)
            x_ext = np.linspace(0, dY.max(), 100)
            p = np.polyfit(SOMLine[:, 0], SOMLine[:, 1], deg=1)
            y_ext = np.poly1d(p)(x_ext)
            lreg.fit(dY.reshape(-1, 1), dX.reshape(-1, 1))
            D = euclidean_dist(y_ext, lreg.coef_[0]*x_ext)
            ax.scatter(dY[K], dX[K], s=1, c='k', marker='s',
                       label='Rate-distortion')
            ax.plot(x_ext, y_ext, 'r', lw=1.5)
            ax.plot(x_ext, lreg.coef_[0]*x_ext, 'm')
            ax.set_xlim([0, dY.max()+0.15])
            ax.set_ylim([0, dX.max()+0.15])
            ax.set_xlabel(r'$\delta{\bf y}$', fontsize=13)
            ax.set_title(r"$\mathcal{P} = $"+str(np.round(D, 2)),
                         fontsize=13, weight='bold')
            if i == 0:
                ax.set_ylabel(r'$\delta{\bf x}$', fontsize=13)
            ax.set_xticks([0, 10, 20])
            ticks = ax.get_xticks()
            ax.set_xticklabels(ticks, fontsize=13, weight='bold')
            ticks = ax.get_yticks()
            ticks = [np.round(i, 2) for i in ticks]
            ax.set_yticklabels(ticks, fontsize=13, weight='bold')
            if i != 0:
                ax.set_yticks([])
        plt.savefig("Figure06.pdf", axis='tight')

    if 0:
        fig = plt.figure(figsize=(20, 3))
        fig.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.2)
        for i, name in enumerate(names):
            ax = fig.add_subplot(1, 8, i+1)
            dist = np.load("./results/distortion_"+name+".npy")
            ax.plot(dist, c='orange', lw=2)
            ax.set_ylim([0, 0.02])
            ax.set_xlim([0, 7000])
            ax.set_xticks([0, 3500, 7000])
            ticks = ax.get_xticks().astype('i')
            ax.set_xticklabels(ticks, fontsize=13, weight='bold')
            ticks = ax.get_yticks()
            ax.set_yticklabels(np.round(ticks, 3), fontsize=13, weight='bold')
            # ax.set_title(cases[i], fontsize=16, weight='bold')
            ax.set_xlabel("Epochs", fontsize=13, weight='bold')
            if i == 0:
                ax.set_ylabel("Distortion", fontsize=15, weight='bold')
            if i != 0:
                ax.set_yticks([])

        plt.savefig("Figure07.pdf", axis='tight')
    plt.show()
