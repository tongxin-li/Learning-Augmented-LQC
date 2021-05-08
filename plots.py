import matplotlib.pyplot as plt


def plot_trajectory(y):

    plt.plot(y[:,0], y[:,1], linewidth=2)


def plot_track(x, y):

    plt.plot(x[:,0]+y[:,0], x[:,1]+y[:,1])

def plot_competitive_ratio(epsilon, competitive_ratio, lam, color):

    plt.plot(epsilon/1000, competitive_ratio, color=color, label=r'$\lambda=$'+str(round(lam,1)))
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')

def plot_upper_bound(epsilon, upper_bound, lam, color):

    plt.plot(epsilon/1000, upper_bound,  color=color, linestyle='dashed', label=r'$\lambda=$'+str(round(lam, 1)))
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')
