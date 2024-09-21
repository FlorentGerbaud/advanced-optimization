#_______________________________________ import modules _______________________________________#

import numpy as np
import math
import matplotlib.pyplot as plt

#_______________________________________ define variables _______________________________________#

# domain length
l = 1
# number of elements
n_el = 200

# element size
h = l/n_el
# number of nodes
n_no = n_el+1
# nodes coordinates
X = np.linspace(0,1,n_no)
# integration points coordinates
Xg = 0.5*( X[:-1] + X[1:] )

# target temperature
T_star = np.ones(X.shape) + (X/l)**2

# initial design vector that parameterizes heat sources
dim_opt = 6
Var_ini = np.full(dim_opt,0)
#Var_ini = np.array([-2,0.1,-2.35,0.25,0.25,0.2])

# conduction coefficient with possible noise

rng = np.random.default_rng(seed=2024)




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

# function that perform the Bernstein polynomial

def performBernstein(t,Var_opt):
    """
    Perform the Bernstein polynomial
    :param t: the point where the polynomial is evaluated
    :param Var_opt: the vector of coefficients
    :return: the value of the polynomial at t
    """
    B = 0
    for i in range(dim_opt):
        B += math.comb(dim_opt-1,i)*t**i*(1-t)**(dim_opt-1-i)*Var_opt[i]
    return B

############################################################################################################
############################################ compute_source ##############################################
############################################################################################################

# function that compute the heat source

def compute_source(Var_opt):
    """
    Compute the heat source
    :param Var_opt: the vector of coefficients
    :return: the heat source
    """
    # by default zero source
    S = np.zeros(Xg.shape)

    #perform the source terms on the interval

    for i in range(n_no-1):
        S[i] = performBernstein(X[i],Var_opt)
    print(S)
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
    # Finite-Element matrix initialisation
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
    # Finite-Element right-hand side initialisation
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

def simulator(noise, Var):
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
    Src = compute_source(Var)

    # right-hand side
    Rhs = compute_rhs(Src)

    # Finite-element solution
    T = np.matmul(np.linalg.inv(M), Rhs).reshape((n_no))

    return T

############################################################################################################
############################################ costFunction #################################################
############################################################################################################

def costFunction():
    """
    Compute the cost function
    :return: the norm of the difference between the initial temperature and the target temperature
    """
    return 0.5*sum((T_ini-T_star)**2)*h



#_______________________________________ main _______________________________________#

# without noise
K_ref = compute_conduction(0)

# solve the problem for initial conditions

T_ini = simulator(0,Var_ini)

# plot of the conduction coefficient along x

plt.plot(Xg,K_ref)
plt.xlabel('x')
plt.ylabel('conduction coefficient')
plt.show()

# plot of initial temperature along x

plt.plot(X,T_ini)
plt.plot(X,T_star,'r--')
plt.xlabel('x')
plt.ylabel('temperature')
plt.legend(["initial", "target"], loc="lower right")
plt.show()
print(performBernstein(0.5,Var_ini))