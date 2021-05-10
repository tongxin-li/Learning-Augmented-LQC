import numpy as np

# Create parameters

def generate_parameters(mode):

    if mode == 'Tracking':

        A = np.array([[1,0,0.2,0],[0,1,0,0.2],[0,0,1,0],[0,0,0,1]])
        B = np.array([[0,0],[0,0],[0.2,0],[0,0.2]])
        Q = np.array([[0.01,0,0,0],[0,0.01,0,0],[0,0,0,0],[0,0,0,0]])
        R = np.array([[0.01,0],[0,0.01]])

        T = 150 # Time steps
        N = 50 # Epsilon divisions
        M = 1 # Number of Monte Carlo tests
        J = 6

        sigma = np.logspace(0,1,N)-1
        sigma = np.linspace(0,1,N)
        lam = np.linspace(0,1,J)
        mu = 0

        return A, B, Q, R, T, N, M, J, sigma, lam, mu

    if mode == 'EV':

        A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        B = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
        Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        R = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        T = 30 # Time steps
        N = 50 # Epsilon divisions
        M = 1 # Number of Monte Carlo tests
        J = 6

        sigma = np.linspace(0,1,N)
        lam = np.linspace(0,1,J)
        mu = 0

        return A, B, Q, R, T, N, M, J, sigma, lam, mu
