#_______________________________________ import modules _______________________________________#

from computeSource import *
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