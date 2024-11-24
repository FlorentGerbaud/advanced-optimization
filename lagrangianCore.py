from costFunction import *

def lagrangianFunction(T_ini, var_opt, dim_opt, beta):
    """
    :param T_ini: initial temperature
    :param var_opt: design variables
    :param dim_opt: number of design variables
    :param beta: Lagrangian coefficient
    :return: the cost function with the energy constraint
    """
    S = compute_source(var_opt, dim_opt)
    integraleofS = np.sum(S) * h
    return costFunction(T_ini) + beta * integraleofS