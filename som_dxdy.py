#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Georgios Is. Detorakis (Georgios.Detorakis@inria.fr)
#
# Implementation of dy-dx representation of self-organizing maps (SOMs)
# according to [1].
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL: http://www.cecill.info.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
#
# Dependencies:
#
#     python > 2.6 (required): http://www.python.org
#     numpy        (required): http://numpy.scipy.org
#     scipy        (required): http://www.scipy.org
#     matplotlib   (required): http://matplotlib.sourceforge.net
#
# -----------------------------------------------------------------------------
# Contributors:
#
#     Georgios Is. Detorakis
#
# Contact Information:
#
#     Georgios Is. Detorakis
#     INRIA Nancy - Grand Est research center
#     CS 20101
#     54603 Villers les Nancy Cedex France
#
# References:
# 1] P. Demartines, "Organization measures and representations of the Kohonen
# maps", First IFIP Working Group 10.6 Workshop, 1992.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pylab as plt
import scipy.spatial.distance as ssp


# Calculates the dy-dx representation of a self-organizing map (SOM).
# Takes as input a matrix contains the weights of a SOM.
# The shape of the matris must be of the form m x n x z.
def som_regularity(data):
    # Initialization of parameters
    m, n, z = data.shape[0], data.shape[1], data.shape[2]
    k = m * n
    size = (k**2 - k)//2

    # R contains the coordinates of the regular grid
    R = np.zeros(data.shape)

    # In dX are stored the distances of the SOM weights
    dX = np.zeros((size,))
    # In dY are stores the distances of the weights regular grid
    dY = np.zeros((size,))
    for i in range(m):
        for j in range(n):
            R[i, j, 0] = i + 1
            R[i, j, 1] = j + 1

    X = data.reshape(k, z)
    Y = R.reshape(k, z)

    # Calculating all possible distances ( (m*n)^2 - (m*n) )/2
    dX = ssp.pdist(X, 'euclidean')
    dY = ssp.pdist(Y, 'euclidean')

    # jj = 0
    # for i in range(k):
    #     for j in range(i+1, k):
    #         dX[jj] = np.sqrt(((X[i] - X[j])**2).sum())
    #         dY[jj] = np.sqrt(((Y[i] - Y[j])**2).sum())
    #         jj += 1

    # Calculating the som line (dX=dY)
    # This line indicates the perfect matching of dX and dY
    # SOMLine = np.arange(size) * dX[0]
    SOMLine = np.array([[0, 0], [dY.mean(), dX.mean()]])
    # SOMLine = np.linspace(-1, 1, size)
    return dX, dY, SOMLine


if __name__ == '__main__':
    # The size of the weights matrix
    n = 16
    m = 2
    # If you want to give the full path of the weight matrix
    # folder = '/home/Local/SOM/Final_Results/32Grid16Receptors5Noise/'
    filename = './data/weights_6999.npy'

    # Read weights from a file
    W = np.load(filename)

    # Calculating the representation dy-dx
    W = W.reshape(n, n, m)
    dX, dY, SOMLine = som_regularity(W)

    # Plotting
    plt.figure()
    plt.scatter(dY, dX, s=1, c='k', marker='s')
    plt.plot(SOMLine, 'r', lw=1.5)
    plt.axis([0, dY.max()+0.15, 0, dX.max()+0.15])
    plt.xlabel('dy')
    plt.ylabel('dx')

    plt.show()
