import numpy as np
import matplotlib.pyplot as plt


epsilon = np.load("e.npy")
online_epsilon = np.load("oe.npy")
competitive_ratio = np.load("cp.npy")
online_competitive_ratio = np.load("ocp.npy")

def plot_competitive_ratio(epsilon, competitive_ratio, lam, color, online):

    if online is True:
        plt.plot(epsilon / 1000, competitive_ratio, color=color, label='Online')
        plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')
    else:
        plt.plot(epsilon/1000, competitive_ratio, color=color, label=r'$\lambda=$'+str(round(lam,1)))
        plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.2, title=r'$\lambda$' + ' Values')


colors = ['blue', 'red', 'green', 'orange', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'skyblue', 'black']
bound_color_index = 0
for k in range(6):
    epsilon[k], competitive_ratio[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], competitive_ratio[k]))))
    plot_competitive_ratio(np.array(epsilon[k]), np.array(competitive_ratio[k]), k * 0.2, colors[bound_color_index],
                           False)
    bound_color_index += 1

online_epsilon, online_competitve_ratio = (list(t) for t in zip(*sorted(zip(online_epsilon, online_competitive_ratio))))
plot_competitive_ratio(np.array(online_epsilon), np.array(online_competitive_ratio), 0, 'black', True)
plt.title("Algorithm Performance")
plt.xlabel('Prediction Error ' + r"$\varepsilon$")
plt.ylabel("Competitive Ratios")
plt.grid()
plt.show()
