import numpy as np

# Matrices

A = np.array([[1,0,0.2,0.2],[0,1,0,0.2],[0,0,1,0],[0,0,0,1]])
B = np.array([[0,0],[0,0],[0.2,0],[0,0.2]])
Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
R = np.array([[1,0],[0,1]])

T = 30 # Time steps
N = 50 # Epsilon divisions
M = 20 # Number of Monte Carlo tests
J = 6

sigma = np.linspace(0,2,N)
lam = np.linspace(0,1,J)

mu = 0