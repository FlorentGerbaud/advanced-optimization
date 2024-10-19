#_______________________________________ import modules _______________________________________#

from variables import *
from computeSource import *

############################################################################################################
############################################ costFunction #################################################
############################################################################################################

def costFunction(T_ini):
    """
    Compute the cost function
    :return: the norm of the difference between the initial temperature and the target temperature
    """
    return 0.5 * sum((T_ini - T_star) ** 2) * h


def augmentedCostFunction(T_ini, var_opt, dim_opt, pen):
    """
    Calculate the cost function J(T) with the energy constraint (integral of S = 0).
    """
    S = compute_source(var_opt, dim_opt)
    # Original cost: difference between current temperature and target
    cost = 0.5 * np.sum((T_ini - T_star) ** 2) * h

    # Compute the energy of the heat source (integral of S(x))
    energy = np.sum(S) * h  # Discrete approximation of the integral of S

    # Add penalty term for energy constraint (integral of S should be 0)
    lambda_c = pen  # Penalty coefficient, you can adjust this value
    cost += lambda_c * energy ** 2

    return cost
