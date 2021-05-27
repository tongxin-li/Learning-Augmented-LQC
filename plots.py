import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np

def plot_lambda(lam):


    opt_lam = lam[-1] * np.ones(len(lam))
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    plt.plot(lam, color='black', label='Self-tuned parameters')
    plt.plot(opt_lam, color='gray', linestyle='dashed', label='Optimal tuning parameter')
    plt.legend(loc='best', scatterpoints=1, frameon=True, labelspacing=0.2, prop={'size': 15})


def plot_trajectory(y,color):

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    plt.plot(y[:,0], y[:,1], linewidth=0.5, color=color, linestyle='dashed', label='Trajectory'+r' $\mathbf{y}$')
    plt.legend(loc='upper right', scatterpoints=1, frameon=True, labelspacing=0.2, title='Trajectories')

def plot_track(x, y,context, color):

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    plt.plot(x[:,0]+y[:,0], x[:,1]+y[:,1], color=color, label=context)
    plt.legend(loc='upper right', scatterpoints=1, frameon=True, labelspacing=0.2, title='Trajectories')

def plot_competitive_ratio(epsilon, competitive_ratio, lam, color, online):

    if online is True:
        plt.plot(epsilon / 1000, competitive_ratio, color=color, label='Online')
        plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')
    else:
        plt.plot(epsilon/1000, competitive_ratio, color=color, label=r'$\lambda=$'+str(round(lam,1)))
        plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')

def plot_upper_bound(epsilon, upper_bound, lam, color):

    plt.plot(epsilon/1000, upper_bound,  color=color, linestyle='dashed', label=r'$\lambda=$'+str(round(lam, 1)))
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')
