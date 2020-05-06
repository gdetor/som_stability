# This script generates Figure No 5 from [1].
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

from condition import condition
from som_dxdy import som_regularity

from sklearn.linear_model import LinearRegression


def euclidean_dist(x, y):
    return np.sqrt(((x - y)**2).sum())


if __name__ == '__main__':
    Ke = [0.47, 0.61, 0.74, 0.8, 0.9, 1.0, 2.0, 3.0]
    Ki = [0.2, 0.3, 0.4, 0.45, 0.53, 0.60, 1.40, 2.25]

    names = ['04702', '01603', '07404', '08045', '09053', '1006', '20135',
             '30225']
    KeKi = zip(Ke, Ki)
    labels = ['('+str(i)+', '+str(j)+')' for i, j in KeKi]

    lreg = LinearRegression(fit_intercept=False)

    cond, mse, dist, reg = [], [], [], []
    for i in range(len(Ke)):
        cond.append(condition(Ke[i], 0.1, Ki[i], 1.0, alpha=0, beta=1))
        print(cond[i])
        # tmp = np.load("./results/new_mse_"+names[i]+".npy")
        # mse.append(tmp[-10:].mean())
        tmp = np.load("results/distortion_"+names[i]+".npy")
        dist.append(tmp[-10:].mean())
        w = np.load("./results/w_hist_"+names[i]+".npy")[-1]
        dX, dY, SOMLine = som_regularity(w.reshape(16, 16, 2))
        x_ext = np.linspace(0, dY.max(), 100)
        p = np.polyfit(SOMLine[:, 0], SOMLine[:, 1], deg=1)
        y_ext = np.poly1d(p)(x_ext)
        lreg.fit(dY.reshape(-1, 1), dX.reshape(-1, 1))
        D = euclidean_dist(y_ext, lreg.coef_[0]*x_ext)
        reg.append(D)
        # reg.append(lreg.coef_[0])
    cond = np.array(cond)
    dist = np.array(dist)
    reg = np.array(reg)

    fig = plt.figure(figsize=(11, 11))
    host = fig.add_subplot(111)
    fig.subplots_adjust(right=0.75, bottom=.2)

    par1 = host.twinx()
    par2 = host.twinx()

    par2.spines["right"].set_position(("axes", 1.2))

    p1, = host.plot(cond, 'k-o', lw=2, ms=10, label="Stability Condition")
    host.axhline(1, c='k', lw=2, ls='--')
    p2, = par1.plot(reg, 'g-', marker='x', lw=2, ms=10,
                    label=r"${\bf \mathcal{P}}$")
    p3, = par2.plot(dist, c='orange', marker='.', lw=2, ms=10,
                    label="Distortion")

    host.set_xlim([-.1, 7])
    host.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    host.set_xticklabels(labels, fontsize=13, weight='bold', rotation=45)
    ticks = host.get_yticks()
    host.set_xlabel(r"$({\bf K_e}, {\bf K_i})$", fontsize=16, weight='bold')
    host.set_yticklabels(np.round(ticks, 2), fontsize=16, weight='bold')

    host.set_xlabel(r"${\bf (K_e, K_i)}$", fontsize=16, weight='bold')
    host.set_ylabel("Stability Condition", fontsize=16, weight='bold')
    par1.set_ylabel(r"${\bf \mathcal{P}}$", fontsize=16,
                    weight='bold')
    par2.set_ylabel("Distortion", fontsize=16, weight='bold')

    ticks = par2.get_yticks()
    par2.set_yticklabels(np.round(ticks, 4), fontsize=16, weight='bold')
    ticks = par1.get_yticks()
    par1.set_yticklabels(np.round(ticks, 4), fontsize=16, weight='bold')

    lines = [p1, p2, p3]
    host.legend(lines, [l.get_label() for l in lines], loc=6)

    for ax in [par1, par2]:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)

        plt.setp(ax.spines.values(), visible=False)
        ax.spines["right"].set_visible(True)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    par1.spines["right"].set_edgecolor(p2.get_color())
    par2.spines["right"].set_edgecolor(p3.get_color())

    host.tick_params(axis='y', colors=p1.get_color())
    par1.tick_params(axis='y', colors=p2.get_color())
    par2.tick_params(axis='y', colors=p3.get_color())

    plt.savefig("Figure05.pdf", axis='tight')
    plt.show()
