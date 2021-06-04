from plots import *
from model import *
from _PARAMETERS import *
import numpy as np
import argparse


def get_configs():

    parser = argparse.ArgumentParser(
        description='Self-tuning Linear Quadratic Controller')

    parser.add_argument('--mode', default='EV', type=str,
                        help='Tracking or EV')
    parser.add_argument('--noise', default='Gaussian', type=str,
                        help='Noise type, Gaussian or Binomial')
    parser.add_argument('--ini_lambda', default=0.3, type=float,
                        help='initial trust parameter lambda, in [0,1]')
    parser.add_argument('--save_output', default=False,
                        type=str, help='Save output or not')
    parser.add_argument('--plot_output', default=True,
                        type=str, help='Plot output or not')
    parser.add_argument('--J', default=6,
                        type=str, help='Number of trust parameters')
    parser.add_argument('--T', default=100,
                        type=str, help='Number of time slots')
    parser.add_argument('--N', default=20,
                        type=str, help='Number of error divisions')
    parser.add_argument('--M', default=5,
                        type=str, help='Number of Monte Carlo tests')


    configs = parser.parse_args()

    return configs


def main():

    configs = get_configs()
    print(configs)

    # Initialize

    mode = configs.mode
    T = configs.T
    N = configs.N
    J = configs.J
    M = configs.M

    A, B, Q, R, sigma, lam, mu = generate_parameters(mode, N, J)

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

            noise = generate_noise(mu, sigma[i], T, A, configs.noise)
            print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:' + str(j))

            # Run self-tuning control
            _epsilon, X, Y, W, Z, _online_ALG, _OPT = run_fix_lqr_robot(T, A, B, Q, R, noise, mode, P,
                                                                                 D, H, F_list, configs.ini_lambda)
            if _OPT != 0 and _online_ALG / _OPT > online_competitive_ratio[i]:
                online_competitive_ratio[i] = _online_ALG / _OPT
                online_epsilon[i] = _epsilon

            for k in range(J):

                print('Runing tests ... ' + 'Epsilon:' + str(i) + ' Monte:' + str(j) + ' Lambda:' + str(k))

                # Run lambda-confident control with different trust patameters
                _myopic_ALG = run_lqr_robot(T, A, B, Q, R, noise, lam[k], mode, P, D, H, F_list)

                if _OPT != 0 and _myopic_ALG / _OPT > competitive_ratio[k, i]:
                    competitive_ratio[k, i] = _myopic_ALG / _OPT
                    epsilon[k, i] = _epsilon
                    # upper_bound[k, i] = compute_upper_bound(A, B, Q, R, _OPT, lam[k], _epsilon, X, Y, W, Z)


    if configs.plot_output:

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

    if configs.save_output:

        # Save data
        np.save("cp.npy", competitive_ratio)
        np.save("ocp.npy", online_competitive_ratio)
        np.save("e.npy", epsilon)
        np.save("oe.npy", online_epsilon)



if __name__ == '__main__':
    main()