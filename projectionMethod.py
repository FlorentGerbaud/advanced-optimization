from performGradient import *
from tqdm import tqdm  # Pour la barre de progression
from costFunction import *
from FEM import *
import numpy as np
import FEM as FiniteElement
import matplotlib
matplotlib.use('Agg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import gradientDescent as gd


def constraintFunction(Var_opt, dim_opt):
    """
    Compute the constraint function
    :param Var_opt: the vector of design variables
    :return: the constraint function
    """
    S = compute_source(Var_opt, dim_opt)
    return sum(S) * h

def constraintGradient(Var_opt, dim_opt):
    """
    Compute the gradient of the constraint function
    :param Var_opt: the vector of design variables
    :return: the gradient of the constraint function
    """
    gradConstraint = np.zeros_like(Var_opt)
    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            #print(type(gradConstraint[i]))
            gradConstraint[i] += math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h
    norm = np.linalg.norm(np.array(gradConstraint))
    projectedGrad = gradConstraint / norm
    normeOfProjectedGrad = np.linalg.norm(projectedGrad)
    return projectedGrad


def solveProjectedGradientMethodWithStepFixed(Var_ini, alpha, iterations, K_ref, dim_opt, T_ini):

    Var_optWithFixStep, errors_fixed_step = gd.gradient_descentProjected(Var_ini, alpha, iterations, K_ref, dim_opt)

    # Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt, pen_fixe)
    TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
    print("Optimized design variables with fixed step: ", Var_optWithFixStep)

    # check energy nulle
    S_check = compute_source(Var_optWithFixStep, dim_opt)
    integralS_x = np.sum(S_check) * h
    print("Integral of the source: ", integralS_x)

    print("Error between Initial and Target Temperature: ", costFunction(TFixedStep))

    # Tracer le graphique log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.savefig("Error With projected.png")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TFixedStep, label='Fixed Step Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Initial, Target and Optimized Temperature')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("temp With projected.png")

    # Plot source term
    plt.figure(figsize=(10, 6))
    plt.plot(Xg, S_check, marker='o')
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.grid()
    plt.savefig("source with projected.png")

def solveProjectedGradientMethodWithStepOptimal(Var_ini, iterations, K_ref, dim_opt, T_ini):

    Var_optWithLineSearch, errors_line_search = gd.gradient_descent_with_line_search(Var_ini,
                                                                                     iterations,
                                                                                     K_ref,
                                                                                     dim_opt)

    # Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt, pen_fixe)
    TOptimalStep = simulator(0, Var_optWithLineSearch, dim_opt)
    print("Optimized design variables with fixed step: ", Var_optWithLineSearch)

    # check energy nulle
    S_check = compute_source(Var_optWithLineSearch, dim_opt)
    integralS_x = np.sum(S_check) * h
    print("Integral of the source: ", integralS_x)

    print("Error between Initial and Target Temperature: ", costFunction(TOptimalStep))

    # Tracer le graphique log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_line_search, marker='o', label='Fixed Step Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.savefig("Error With projected.png")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TOptimalStep, label='Fixed Step Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Initial, Target and Optimized Temperature')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("temp With projected.png")

    # Plot source term
    plt.figure(figsize=(10, 6))
    plt.plot(Xg, S_check, marker='o')
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.grid()
    plt.savefig("source with projected.png")













