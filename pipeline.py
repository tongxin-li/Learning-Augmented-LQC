from plots import *
from model import *
from _PARAMETERS import *


# Initialize

mode = 'Tracking'

A, B, Q, R, T, N, M, J, sigma, lam, mu = generate_parameters(mode)

competitive_ratio = np.zeros((J,N))
online_competitive_ratio = np.zeros(N)
upper_bound = np.zeros((J,N))
epsilon = np.zeros((J,N))
online_epsilon = np.zeros(N)

for i in range(N):

    for j in range(M):

        noise = generate_noise(mu, sigma[i], T, A)

        for k in range(J):

            print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:'+ str(j) + ' Lambda:' + str(k))
            _epsilon, X, Y, W, Z, _myopic_ALG, _online_ALG ,_OPT = run_robot(T, A, B, Q, R, noise, lam[k], mode)

            if _OPT != 0 and _myopic_ALG/_OPT > competitive_ratio[k,i]:

                competitive_ratio[k,i] = _myopic_ALG/_OPT
                epsilon[k,i] = _epsilon
                upper_bound[k,i] = compute_upper_bound(A, B, Q, R, _OPT, lam[k], _epsilon,  X, Y, W, Z)

            if _OPT != 0 and _online_ALG / _OPT > online_competitive_ratio[i]:

                online_competitive_ratio[i] = _online_ALG / _OPT
                online_epsilon[i] = _epsilon

# Plotting

colors = ['blue', 'red', 'green', 'orange', 'gray', 'brown', 'cyan', 'magenta', 'yellow','skyblue', 'black']
bound_color_index = 0

for k in range(J):

    _, upper_bound[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], upper_bound[k]))))
    epsilon[k], competitive_ratio[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], competitive_ratio[k]))))
    plot_competitive_ratio(np.array(epsilon[k]), np.array(competitive_ratio[k]), lam[k], colors[bound_color_index], False)
    # plot_upper_bound(np.array(epsilon[k]), upper_bound[k], lam[k], colors[bound_color_index])
    bound_color_index += 1

plot_competitive_ratio(np.array(online_epsilon), np.array(online_competitive_ratio), 0, 'black', True)

plt.title("Algorithm Performance")
plt.xlabel('Prediction Error '+r"$\varepsilon$")
plt.ylabel("Competitive Ratios")
plt.grid()
plt.show()