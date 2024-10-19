#______________________________________ import libraries __________________________________________________

from costFunction import *
from FEM import *
import math
from variables import *

#_______________________________________ define functions __________________________________________________

############################################################################################################
############################################ compute_adjoint ###############################################
############################################################################################################

def compute_adjoint(T, T_star, K):
    """
    Solve the adjoint equation to compute the Lagrange multiplier lambda
    :param T: the computed temperature
    :param T_star: the target temperature
    :param K: the conduction coefficient
    :return: the Lagrange multiplier lambda
    """
    # Finite-Element matrix initialization
    M_adj = np.zeros((n_no, n_no))
    Rhs_adj = np.zeros((n_no, 1))

    # Boundary conditions for the adjoint problem (Dirichlet boundary conditions, lambda = 0 at boundaries)
    M_adj[0][0] = 1
    M_adj[n_no - 1][n_no - 1] = 1
    # boundary conditions allwos to have lambda = 0 at boundaries
    # so l(x).k(x).grad(T) = 0 at boundaries
    #and d/dx(l(x).k(x).T) = 0 at boundaries
    Rhs_adj[0] = 0
    Rhs_adj[n_no - 1] = 0

    # Internal coefficients
    for i in range(1, n_no - 1):
        M_adj[i][i] = (K[i - 1] + K[i]) / h
        M_adj[i][i - 1] = -K[i - 1] / h
        M_adj[i][i + 1] = -K[i] / h
        #Rhs_adj[i] = 2 * (T[i] - T_star[i])  # RHS of the adjoint equation
        Rhs_adj[i] = - (T[i] - T_star[i]) * h # RHS of the adjoint equation

    # Solve the adjoint equation
    lambda_adj = np.matmul(np.linalg.inv(M_adj), Rhs_adj).reshape((n_no))

    return lambda_adj

############################################################################################################
############################################ compute_gradient ##############################################
############################################################################################################

def compute_gradient(lambda_adj, Var_opt, dim_opt):
    """
    Compute the gradient of the cost function with respect to the design variables Var_opt
    :param lambda_adj: the Lagrange multiplier
    :param Var_opt: the vector of design variables
    :return: the gradient of the cost function
    """
    grad = np.zeros(dim_opt)

    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            grad[i] += - lambda_adj[j] * math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h

    return grad

def finite_difference_gradient_centered(Var_opt, epsilon, dim_opt):
    """
    Compute the gradient of the cost function using finite differences with centered scheme
    :param Var_opt: the vector of design variables
    :param epsilon: the perturbation value
    :return: the gradient of the cost function
    """
    grad = np.zeros(dim_opt)

    for i in range(dim_opt):
        Var_perturbed = Var_opt.copy()
        Var_perturbed[i] += epsilon
        T_perturbed = simulator(0, Var_perturbed, dim_opt)
        error_perturbed = costFunction(T_perturbed)

        Var_perturbed[i] -= 2 * epsilon
        T_perturbed = simulator(0, Var_perturbed, dim_opt)
        error_perturbed -= costFunction(T_perturbed)

        grad[i] = error_perturbed / (2 * epsilon)

    return grad


def augmentedCompute_gradient(lambda_adj, Var_opt, dim_opt, pen):
    """
    Compute the gradient of the cost function with respect to Var_opt, including the energy constraint term.
    """
    S = compute_source(Var_opt, dim_opt)  # Heat source
    # Original gradient: using the adjoint solution
    grad = np.zeros_like(Var_opt)
    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            grad[i] += - lambda_adj[j] * math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h

    # Energy constraint gradient (additional term)
    energy_grad = np.sum(S) * h  # The energy constraint term
    lambda_c = pen  # Penalty coefficient
    energy_penalty_grad = 2 * lambda_c * energy_grad  # Derivative of energy penalty term

    # Adjust the gradient
    grad += energy_penalty_grad

    return grad
