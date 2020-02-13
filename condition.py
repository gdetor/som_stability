# This script checks if the stability condition introdiced in [1] holds true
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
from scipy.special import erf


def xi(sigma, a=-1, b=1):
    tmp = (2 * sigma**2 * (np.exp(-(a - b)**2 / (2 * sigma**2)) - 1)
           + sigma * np.sqrt(2*np.pi) * (a - b)
           * erf((a - b) / (sigma * np.sqrt(2))))
    return tmp


def condition(x):
    Ke = x[0]
    sigma_e = 0.1
    Ki = x[1]
    sigma_i = 1.0
    res = (Ke**2 * xi(sigma_e * np.sqrt(2)) + Ki**2 * xi(sigma_i * np.sqrt(2))
           - 2 * Ke * Ki
           * xi((sigma_e * sigma_i) / np.sqrt(sigma_e**2 + sigma_i**2)))
    return res


if __name__ == '__main__':
    x_stable = np.array([0.9, 0.5])
    x_unstable = np.array([1.3, 0.7])
    res = condition(x_unstable)
    if res < 1:
        print("Stable equilibrium: %f < 1" % (res))
    else:
        print("Unstable equilibrium: %f > 1" % (res))
