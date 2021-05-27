from plots import *
from model import *
from _PARAMETERS import *
import numpy as np

# Initialize

mode = 'EV'

A, B, Q, R, T, N, M, J, sigma, lam, mu = generate_parameters(mode)
P, _, _ = control.dare(A, B, Q, R)
D = np.matmul(np.linalg.inv(R + np.matmul(np.matmul(np.transpose(B), P), B)), np.transpose(B))
H = np.matmul(B, D)
F = A - np.matmul(H, np.matmul(P, A))
F_list = [np.linalg.matrix_power(F, i) for i in range(T + 1)]

competitive_ratio = np.zeros((J, N))
online_competitive_ratio = np.zeros(N)
upper_bound = np.zeros((J, N))
epsilon = np.zeros((J, N))
online_epsilon = np.zeros(N)

for i in range(N):

    for j in range(M):

        noise = generate_noise(mu, sigma[i], T, A)
        print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:' + str(j))

        _epsilon, X, Y, W, Z, _online_ALG, _OPT = run_fix_lqr_robot(T, A, B, Q, R, noise, mode, P,
                                                                             D, H, F_list)
        if _OPT != 0 and _online_ALG / _OPT > online_competitive_ratio[i]:
            online_competitive_ratio[i] = _online_ALG / _OPT
            online_epsilon[i] = _epsilon

        for k in range(J):

            print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:' + str(j) + ' Lambda:' + str(k))
            _myopic_ALG = run_lqr_robot(T, A, B, Q, R, noise, lam[k], mode, P, D, H, F_list)

            if _OPT != 0 and _myopic_ALG / _OPT > competitive_ratio[k, i]:
                competitive_ratio[k, i] = _myopic_ALG / _OPT
                epsilon[k, i] = _epsilon
                upper_bound[k, i] = compute_upper_bound(A, B, Q, R, _OPT, lam[k], _epsilon, X, Y, W, Z)


# Plotting

colors = ['blue', 'red', 'green', 'orange', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'skyblue', 'black']
bound_color_index = 0

for k in range(J):
    _, upper_bound[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], upper_bound[k]))))
    epsilon[k], competitive_ratio[k] = (list(t) for t in zip(*sorted(zip(epsilon[k], competitive_ratio[k]))))
    plot_competitive_ratio(np.array(epsilon[k]), np.array(competitive_ratio[k]), lam[k], colors[bound_color_index],False)
    # plot_upper_bound(np.array(epsilon[k]), upper_bound[k], lam[k], colors[bound_color_index])
    bound_color_index += 1

online_epsilon, online_competitve_ratio = (list(t) for t in zip(*sorted(zip(online_epsilon, online_competitive_ratio))))
plot_competitive_ratio(np.array(online_epsilon), np.array(online_competitive_ratio), 0, 'black', True)

plt.title("Algorithm Performance")
plt.xlabel('Prediction Error ' + r"$\varepsilon$")
plt.ylabel("Competitive Ratios")
# plt.ylabel("Upper Bounds")
plt.grid()
plt.show()

# np.save("cp.npy", competitive_ratio)
# np.save("ocp.npy", online_competitive_ratio)
# np.save("e.npy", epsilon)
# np.save("oe.npy", online_epsilon)