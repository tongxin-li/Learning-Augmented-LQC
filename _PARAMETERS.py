import numpy as np

# Matrices

A = np.array([[1,0,0.2,0.2],[0,1,0,0.2],[0,0,1,0],[0,0,0,1]])
B = np.array([[0,0],[0,0],[0.2,0],[0,0.2]])
Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
R = np.array([[0.01,0],[0,0.01]])

T = 30 # Time steps
N = 50 # Epsilon divisions
M = 10 # Number of Monte Carlo tests
J = 6

sigma = np.logspace(0,1,N)-1
sigma = np.linspace(0,1,N)
lam = np.linspace(0,1,J)
mu = 0

mode = 'Tracking' # Options: Tracking, Gaussian, EV