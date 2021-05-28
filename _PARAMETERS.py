import numpy as np


# Create parameters

def generate_parameters(mode):
    if mode == 'Tracking':
        A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
        Q = np.array([[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        R = np.array([[0.01, 0], [0, 0.01]])

        T = 200  # Time steps
        N = 30  # Epsilon divisions
        M = 5  # Number of Monte Carlo tests
        J = 6

        sigma = np.logspace(0, 1, N) - 1
        sigma = np.linspace(0, 1, N)
        lam = np.linspace(0, 1, J)
        # lam = np.array([0, 0.2, 0.4])
        mu = 0

        return A, B, Q, R, T, N, M, J, sigma, lam, mu

    if mode == 'EV':

        A = np.eye(10)
        B = -np.eye(10)
        Q = np.eye(10)
        R = 0.1*np.eye(10)

        T = 100 # Time steps
        N = 20 # Epsilon divisions
        M = 5 # Number of Monte Carlo tests
        J = 6

        sigma = np.linspace(0,10,N)
        lam = np.linspace(0,1,J)
        mu = 0

        return A, B, Q, R, T, N, M, J, sigma, lam, mu