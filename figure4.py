import numpy as np
import matplotlib.pylab as plt


def plot_mse():
    mse_stable = np.load("./data/mse_stable.npy")
    mse_unstable = np.load("./data/mse_unstable.npy")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(mse_stable, 'k', lw=2, label='Stable SOM', zorder=10, alpha=.8)
    ax.plot(mse_unstable, 'b', lw=2, label='Unstable SOM', zorder=0,
            alpha=.5)
    ax.set_xlim([0, 7000])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Epochs", fontsize=15, weight='bold')
    ax.set_ylabel("MSE", fontsize=15, weight='bold')
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=15, weight='bold')
    ticks = [np.round(i, 2) for i in ax.get_yticks()]
    ax.set_yticklabels(ticks, fontsize=15, weight='bold')
    ax.legend()


if __name__ == '__main__':
    plot_mse()
    plt.savefig("Figure04.pdf", axis='tight')
    plt.show()
