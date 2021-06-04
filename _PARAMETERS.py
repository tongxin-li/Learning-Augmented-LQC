import numpy as np


# Create parameters

def generate_parameters(mode, N, J):

    if mode == 'Tracking':
       # A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        A = np.array([[0, 0, 10, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [100, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
        Q = np.array([[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        R = np.array([[0.01, 0], [0, 0.01]])

        sigma = np.linspace(0, 1, N)
        lam = np.linspace(0, 1, J)
        mu = 0

        return A, B, Q, R, sigma, lam, mu

    if mode == 'EV':

        A = np.eye(10)
        B = -np.eye(10)
        Q = np.eye(10)
        R = 0.1*np.eye(10)

        sigma = np.linspace(0,10,N)
        lam = np.linspace(0,1,J)
        mu = 0

        return A, B, Q, R, sigma, lam, mu