#_______________________________________ import modules _______________________________________#

from variables import *
import math
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé

#_______________________________________ define functions _______________________________________#

############################################################################################################
############################################ Compute_conduction ############################################
############################################################################################################

# function that compute the conduction coefficient with possible noise

def compute_conduction(noise_val):
    """
    Compute the conduction coefficient with possible noise
    :param noise_val: the noise value
    :return: the conduction coefficient
    """
    noise = rng.random(Xg.shape) * noise_val
    K = np.ones(Xg.shape) + 0.4 * np.sin(4 * np.pi * Xg / l) + 0.3 * np.sin(12 * np.pi * Xg / l) + noise
    return K

############################################################################################################
############################################ performBernstein #############################################
############################################################################################################

# function that performs the Bernstein polynomial

def performBernstein(t, Var_opt, dim_opt):
    """
    Perform the Bernstein polynomial
    :param t: the point where the polynomial is evaluated
    :param Var_opt: the vector of coefficients
    :return: the value of the polynomial at t
    """
    B = 0
    for i in range(dim_opt):
        B += math.comb(dim_opt - 1, i) * t ** i * (1 - t) ** (dim_opt - 1 - i) * Var_opt[i]
    return B

############################################################################################################
############################################ compute_source ##############################################
############################################################################################################

# function that computes the heat source

def compute_source(Var_opt, dim_opt):
    """
    Compute the heat source
    :param Var_opt: the vector of coefficients
    :return: the heat source
    """
    # by default zero source
    S = np.zeros(Xg.shape)

    # perform the source terms on the interval
    for i in range(Xg.shape[0]):
        S[i] = performBernstein(Xg[i], Var_opt, dim_opt)
    return S

############################################################################################################
############################################ compute_matrix ################################################
############################################################################################################

def compute_matrix(K):
    """
    Compute the matrix of the problem
    :param K: the conduction coefficient
    :return: the matrix of the problem
    """
    # Finite-Element matrix initialization
    M = np.zeros((n_no, n_no))

    # Boundary conditions
    M[0][0] = 1
    M[n_no - 1][n_no - 1] = 1

    # Internal coeff
    for i in range(1, n_no - 1):
        M[i][i] = (K[i - 1] + K[i]) / h
        M[i][i - 1] = -K[i - 1] / h
        M[i][i + 1] = -K[i] / h

    return M

############################################################################################################
############################################ compute_rhs ###################################################
############################################################################################################

def compute_rhs(S):
    """
    Compute the right-hand side of the problem
    :param S: the heat source
    :return: the right-hand side of the problem
    """
    # Finite-Element right-hand side initialization
    Rhs = np.zeros((n_no, 1))

    # Boundary conditions
    Rhs[0] = 1
    Rhs[n_no - 1] = 2

    # internal coeff
    for i in range(1, n_no - 1):
        Rhs[i] = (S[i - 1] + S[i]) * h / 2

    return Rhs

############################################################################################################
############################################ simulator #####################################################
############################################################################################################

def simulator(noise, Var, dim_opt):
    """
    Simulate the problem
    :param noise: The noise value
    :param Var: the vector of coefficients
    :return: the temperature
    """
    # conduction
    K = compute_conduction(noise)

    # matrix
    M = compute_matrix(K)

    # compute heat source
    Src = compute_source(Var, dim_opt)

    # right-hand side
    Rhs = compute_rhs(Src)

    # Finite-element solution
    T = np.matmul(np.linalg.inv(M), Rhs).reshape((n_no))

    return T

############################################################################################################
############################################ costFunction #################################################
############################################################################################################

def costFunction(T_ini):
    """
    Compute the cost function
    :return: the norm of the difference between the initial temperature and the target temperature
    """
    return 0.5 * sum((T_ini - T_star) ** 2) * h