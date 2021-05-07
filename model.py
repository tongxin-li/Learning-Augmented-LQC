import numpy as np
import math
import control
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from plots import *

# Define trackings

def tracking_coordinates(t):

    y_1 = 16 * np.power(math.sin(t / 4),3)
    y_2 = 13 * math.cos(t / 4) - 5 * math.cos(2 * t / 4) - 2 * math.cos(3 * t / 4) - math.cos(t)

    return y_1, y_2

# Matrix calculations

def _get_D(B,P,R):

    D = np.matmul(np.linalg.inv(R+np.matmul(np.matmul(np.transpose(B),P),B)),np.transpose(B))

    return D

def _get_H(B,D):

    H = np.matmul(B,D)

    return H

def _get_F(A,P,H):

    F = A - np.matmul(H,np.matmul(P,A))

    return F

def _get_K(F,P,H):

    K = np.linalg.inv(P) - np.matmul(F, np.matmul(np.linalg.inv(P),np.transpose(F))) - H

    return K

def compute_upper_bound(A, B, Q, R, OPT, lam, epsilon, W, Z):

    P, _, _ = control.dare(A, B, Q, R)
    D = _get_D(B, P, R)
    H = _get_H(B, D)
    F = _get_F(A, P, H)

    # K = _get_K(F,P,H)
    # w, _ = np.linalg.eig(K)
    # lam_min = min(w)
    #
    # if lam_min > 0:
    #
    #     bound_1 = 1 + np.linalg.norm(H,2) * (lam*epsilon/OPT + (1-lam)/lam_min)
    #     bound_2 = 1 + np.linalg.norm(H,2) * (1/lam_min + lam*W/OPT)

    bound_1 = 1 + np.linalg.norm(H,2) * (lam*epsilon/OPT + Z*(1-lam)/OPT)
    bound_2 = 1 + np.linalg.norm(H,2) * (Z/OPT + lam*W/(OPT))

    return min(bound_1, bound_2)

def generate_noise(mu, sigma, T, A):

    noise = np.zeros((T, np.shape(A)[0]))

    for t in range(T):

        noise[t] = np.random.normal(mu, sigma, np.shape(A)[0])
        noise[t] = np.random.binomial(5, sigma, np.shape(A)[0])

    return noise

def generate_w(mode, T, A):

    w = np.zeros((T, np.shape(A)[0]))

    if mode == 'Tracking':

        y = np.zeros((T, 2))

        for t in range(T):

            y_1,y_2 = tracking_coordinates(t)
            y_3,y_4 = tracking_coordinates(t+1)
            y[t] = [y_1,y_2]

            # Ground-true predictions
            w[t] = np.matmul(A,np.array([y_1,y_2,0,0])) - np.matmul(A,np.array([y_3,y_4,0,0]))

    if mode == 'Gaussian':

        mu = 0
        sigma = 1

        for t in range(T):

            w[t] = np.random.normal(mu, sigma, np.shape(A)[0])

    return w

def run_robot(T,A,B,Q,R,w,noise,lam):

    # Initialize

    _optimal_u = 0
    x = np.zeros((T, np.shape(A)[0]))
    _optimal_x = np.zeros((T, np.shape(A)[0]))
    estimated_w = np.zeros((T, 4))
    W = 0
    Z = 0
    epsilon = 0

    P, _, _ = control.dare(A, B, Q, R)
    D = _get_D(B, P, R)
    H = _get_H(B, D)
    F = _get_F(A, P, H)

    ALG = 0
    OPT = 0

    for t in range(T):

        # Compute norms
        for s in range(t,T):

            estimated_w[t] = w[t] + noise[t]
            epsilon += np.linalg.norm(matrix_power(F,s-t),2) * np.linalg.norm(P,2) * np.linalg.norm(noise)
            W += np.linalg.norm(matrix_power(F,s-t),2) * np.linalg.norm(P,2) * np.linalg.norm(estimated_w[t])
            Z += np.linalg.norm(matrix_power(F,s-t),2) * np.linalg.norm(P,2) * np.linalg.norm(w[t])

    for t in range(T):

        # Update actions

        E = np.matmul(P,np.matmul(A,x[t]))
        G = 0
        _optimal_E = np.matmul(P,np.matmul(A,_optimal_x[t]))
        _optimal_G = 0

        for s in range(T-t):
            G += np.matmul(np.linalg.matrix_power(np.transpose(F),s), np.matmul(P,estimated_w[s+t]))
            _optimal_G += np.matmul(np.linalg.matrix_power(np.transpose(F),s), np.matmul(P,w[s+t]))

        u = -np.matmul(D,E) - lam * np.matmul(D,G)
        _optimal_u = -np.matmul(D,_optimal_E) - np.matmul(D,_optimal_G)

        # Update states

        if t < T-1:
            x[t+1] = np.matmul(A, x[t]) + np.matmul(B,u) + w[t]
            _optimal_x[t + 1] = np.matmul(A, _optimal_x[t]) + np.matmul(B, _optimal_u) + w[t]

        # Update costs

        if t < T-1:

            ALG += (x[t][0] ** 2) + (x[t][1] ** 2) + 0.01*(u[0] ** 2) + 0.01*(u[1] ** 2)
            OPT += (_optimal_x[t][0] ** 2) + 0.01*(_optimal_x[t][1] ** 2) + 0.01*(_optimal_u[0] ** 2) + (_optimal_u[1] ** 2)
        else:
            ALG += np.matmul(np.transpose(x[t]),np.matmul(P,x[t]))
            OPT += np.matmul(np.transpose(_optimal_x[t]),np.matmul(P,_optimal_x[t]))

    # y = np.zeros((T, 2))
    #
    # for t in range(T):
    #     y_1, y_2 = tracking_coordinates(t)
    #     y_3, y_4 = tracking_coordinates(t + 1)
    #     y[t] = [y_1, y_2]
    # plot_track(x,y)
    # plot_trajectory(y)
    # plt.grid()
    # plt.show()

    print("Algorithm Cost is")
    print(ALG)
    print("Optimal Cost is")
    print(OPT)
    return epsilon, W, Z, ALG, OPT