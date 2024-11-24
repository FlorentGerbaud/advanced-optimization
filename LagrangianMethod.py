#______________________________ import libraries ______________________________

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gradientDescent as gd
import performGradient as pg
from FEM import *
from costFunction import *
import os
import time
from lagrangianCore import lagrangianFunction

#_______________________________________ define functions ______________________________

##################################### lagrangianFunction #####################################
#_____________________________________________________________________________________________


def comparePerformedGradient(dim_opt, K_ref, beta):
    # Compute the gradient with the two methods
    T = simulator(0, Var_ini, dim_opt)
    lambda_adj = pg.compute_adjoint(T, T_star, K_ref)
    grad = pg.lagrangianCompute_gradientByVarOpt(lambda_adj, Var_ini, dim_opt, beta)
    #grad = augmentedGradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt, pen)
    grad_fd = pg.lagrangien_finite_difference_gradient_centered(Var_ini, 1e-6, dim_opt, beta)

    print(f"Gradient with fixed step: {grad}")
    print(f"Gradient with optimal step: {grad_fd}")

    # Compute the difference between the two gradients
    diff = np.linalg.norm(grad - grad_fd)
    print(f"Difference between the two gradients: {diff}")

    # plot the two gradients
    plt.figure(figsize=(10, 6))
    plt.plot(range(dim_opt), grad, marker='o', label='Gradient with Fixed Step')
    plt.plot(range(dim_opt), grad_fd, marker='s', label='Gradient with Optimal Step')
    plt.xlabel('Design Variables')
    plt.ylabel('Gradient')
    plt.title('Comparison of Gradients')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def findBestAlpha_1(Var_ini, alpha_1, alpha_2, iterations, K_ref, dim_opt, beta):

    # find best alpha fixing beta and making a grid of alpha
    alpha_values = np.linspace(0, 20, 100)
    errors = []
    for alpha_1 in alpha_values:
        Var_optWithFixStep, errors_fixed_step = gd.lagragianGradient_descent(Var_ini,
                                                                          alpha_1,
                                                                          alpha_2,
                                                                          iterations,
                                                                          K_ref,
                                                                          dim_opt,
                                                                          beta)
        TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
        errors.append(costFunction(TFixedStep))

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, errors, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.title('Error as a function of Alpha')
    plt.grid()

    # Chemin pour sauvegarder la figure dans le répertoire des résultats
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\graphs\lagrangian_method"
    os.makedirs(results_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas

    # Sauvegarder la figure dans le répertoire spécifié
    fig_path = os.path.join(results_dir, "error_vs_learningRateForALpha_2.png")
    plt.savefig(fig_path)

    print(f"Figure sauvegardée dans : {fig_path}")


def findbestAlpha_2(Var_ini, alpha_1, alpha_2, iterations, K_ref, dim_opt, beta):

    # find best beta fixing alpha and making a grid of beta
    alpha_values = np.linspace(0, 0.01, 100)
    errors = []
    for alpha_2 in alpha_values:
        Var_optWithFixStep, errors_fixed_step = gd.lagragianGradient_descent(Var_ini,
                                                                          alpha_1,
                                                                          alpha_2,
                                                                          iterations,
                                                                          K_ref,
                                                                          dim_opt,
                                                                          beta)
        TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
        errors.append(costFunction(TFixedStep))

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, errors, marker='o')
    plt.xlabel('Beta')
    plt.ylabel('Error')
    plt.title('Error as a function of Beta')
    plt.grid()

    # Chemin pour sauvegarder la figure dans le répertoire des résultats
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\graphs\lagrangian_method"
    os.makedirs(results_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas

    # Sauvegarder la figure dans le répertoire spécifié
    fig_path = os.path.join(results_dir, "error_vs_learningRateForALpha_2.png")
    plt.savefig(fig_path)

    print(f"Figure sauvegardée dans : {fig_path}")




def solveLagrangienFunction(Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, beta, alpha_1, alpha_2):
    # Définir le nom du répertoire pour cette méthode
    dir_name = f"graphs/lagrangian_method/fixed Step/{iterations}_iterations"
    os.makedirs(dir_name, exist_ok=True)  # Crée le répertoire si nécessaire

    # Lancer le chronomètre pour mesurer le temps de traitement
    start_time = time.time()

    # Optimisation avec la méthode du gradient de Lagrangien
    Var_optWithFixStep, errors_fixed_step = gd.lagragianGradient_descent(Var_ini, alpha_1, alpha_2, iterations, K_ref, dim_opt, beta)
    TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
    print("Optimized design variables with fixed step: ", Var_optWithFixStep)

    # Calcul de l'intégrale de la source
    S_check = compute_source(Var_optWithFixStep, dim_opt)
    integralS_x = np.sum(S_check) * h
    error_fixed_step = costFunction(TFixedStep)

    # Mesurer le temps de traitement
    time_fixed_step = time.time() - start_time

    # Tracé log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.savefig(os.path.join(dir_name, "error_fixed_step.png"))

    # Tracer la température
    plt.figure(figsize=(12, 6))
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TFixedStep, label='Fixed Step Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Initial, Target and Optimized Temperature')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(dir_name, "temperature_comparison.png"))

    # Tracer le terme source
    plt.figure(figsize=(10, 6))
    plt.plot(Xg, S_check, marker='o')
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.grid()
    plt.savefig(os.path.join(dir_name, "source_term.png"))

    # Définir le nom du répertoire pour cette méthode
    method_name = "lagrangian"
    dir_name = os.path.join(r"C:\Users\flore\PycharmProjects\advanced-optimization\results",
                            f"{method_name}_method", f"{iterations}_iterations")
    os.makedirs(dir_name, exist_ok=True)  # Crée le répertoire si nécessaire

    res_file_name = os.path.join(dir_name, "LagrangianMethod.res")
    with open(res_file_name, "w") as file:
        file.write(f"Optimized design variables with fixed step: {Var_optWithFixStep}\n")
        file.write(f"Processing Time (Fixed Step): {time_fixed_step:.2f} seconds\n")
        file.write(f"Integral of the source: {integralS_x:.5e}\n")
        file.write(f"Error between Initial and Target Temperature: {costFunction(T_ini):.5f}\n")
        file.write(f"Error between Fixed Step Optimization and Target Temperature: {error_fixed_step:.5f}\n")
        file.write(f"Total Processing Time: {time_fixed_step:.2f} seconds\n")

    print(f"Résultats sauvegardés dans le répertoire : {dir_name}")

    return TFixedStep, S_check, errors_fixed_step

