import os
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error


from condition import condition
from plot_functions import plot_weights


def load_weights():
    list_dir = os.listdir('./data/params/')
    list_dir = [f.lower() for f in list_dir]
    sorted_list = sorted(list_dir)
    data = []
    for name in sorted_list:
        data.append(np.load('./data/params/'+name))
    return np.array(data)


def plot_parameters():
    w = load_weights()
    w_ref = np.load('./data/new_weight_final.npy')

    exc = [np.round(0.4 + i/30, 3) for i in range(30)]

    n = len(exc)
    R = np.ones((n, ))
    C = np.ones((n, ))
    ii = 0
    for i in range(n):
        if np.count_nonzero(np.isnan(w[ii])) == 0:
            R[i] = mean_squared_error(w[ii], w_ref)
            x = np.array([exc[i], 0.2])
            C[i] = condition(x)
            print(ii, exc[i], R[i], C[i], np.linalg.norm(w[ii], ord=2))
        ii += 1

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(121)
    ax.plot(R, '-x', color='k', lw=2)
    ax.plot(C, '-x', color='r', lw=2)

    ax = fig.add_subplot(122)
    samples = np.random.uniform(-1, 1, (100, 2))
    plot_weights(samples, w[7].reshape(16, 16, 2), ax)

    fig = plt.figure()
    plt.hist(w[7], bins=20)


if __name__ == '__main__':
    plot_parameters()
    plt.show()
