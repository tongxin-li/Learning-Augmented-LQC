from plots import *
from model import *
from _PARAMETERS import *


# Initialize

mode = 'Tracking'

A, B, Q, R, T, N, M, J, sigma, lam, mu = generate_parameters(mode)

competitive_ratio = np.zeros((J,N))
upper_bound = np.zeros((J,N))
epsilon = np.zeros((J,N))

for i in range(N):

    for j in range(M):

        noise = generate_noise(mu, sigma[i], T, A)

        for k in range(J):

            print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:'+ str(j) + ' Lambda:' + str(k))
            _epsilon, X, Y, W, Z,_ALG, _OPT = run_robot(T, A, B, Q, R, noise, lam[k], mode)

            if _OPT != 0 and _ALG/_OPT > competitive_ratio[k,i]:

                competitive_ratio[k,i] = _ALG/_OPT
                epsilon[k,i] = _epsilon
                upper_bound[k,i] = compute_upper_bound(A, B, Q, R, _OPT, lam[k], _epsilon,  X, Y, W, Z)

# Plotting

colors = ['blue', 'red', 'green', 'orange', 'gray', 'brown', 'cyan', 'magenta', 'yellow','skyblue', 'black']
bound_color_index = 0

for k in range(J):

    _, upper_bound[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], upper_bound[k]))))
    epsilon[k], competitive_ratio[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], competitive_ratio[k]))))
    # plot_competitive_ratio(np.array(epsilon[k]), np.array(competitive_ratio[k]), lam[k], colors[bound_color_index])
    plot_upper_bound(np.array(epsilon[k]), upper_bound[k], lam[k], colors[bound_color_index])
    bound_color_index += 1

plt.title("Algorithm Performance")
plt.xlabel('Prediction Error '+r"$\varepsilon$")
plt.ylabel("Competitive Ratios")
plt.grid()
plt.show()