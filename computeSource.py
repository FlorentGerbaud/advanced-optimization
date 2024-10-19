#_______________________________________ import modules _______________________________________#

from variables import *
import math
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé

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