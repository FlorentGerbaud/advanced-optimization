from performGradient import *
from tqdm import tqdm  # Pour la barre de progression
from costFunction import *
from FEM import *
import numpy as np
import FEM as FiniteElement
import matplotlib
matplotlib.use('Agg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé
import matplotlib.pyplot as plt
import os
import time
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



def solveProjectedGradientMethodWithStepFixed(Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, method_name="projected"):

    # Définir le nom du répertoire pour cette méthode
    dir_name = f"graphs/{method_name}_method/fixed Step/{iterations}_iterations"
    os.makedirs(dir_name, exist_ok=True)  # Crée le répertoire si nécessaire

    # Lancer le chronomètre pour mesurer le temps de traitement
    start_time = time.time()

    # Optimisation avec pas fixe
    Var_optWithFixStep, errors_fixed_step = gd.gradient_descentProjected(Var_ini, alpha, iterations, K_ref, dim_opt)
    TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)

    # Enregistrer les informations de la variable optimisée
    S_check = compute_source(Var_optWithFixStep, dim_opt)
    integralS_x = np.sum(S_check) * h
    error_fixed_step = costFunction(TFixedStep)

    # Mesurer le temps de traitement pour l'optimisation à pas fixe
    time_fixed_step = time.time() - start_time

    # Tracer et sauvegarder les graphiques
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.savefig(f"{dir_name}/error_{method_name}.png")

    plt.figure(figsize=(12, 6))
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TFixedStep, label='Fixed Step Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Initial, Target and Optimized Temperature')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{dir_name}/temp_{method_name}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(Xg, S_check, marker='o')
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.grid()
    plt.savefig(f"{dir_name}/source_{method_name}.png")

    # Créer le chemin complet pour le fichier .res
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\results"
    file_name = os.path.join(results_dir, f"{method_name}_method", "fixed_step", f"{iterations}_iterations",
                             "ProjectedMethod.res")

    # Créer les répertoires parents si ils n'existent pas
    os.makedirs(os.path.dirname(file_name), exist_ok=True)


    with open(file_name, "w") as file:
        file.write(f"Optimized design variables with line search step: {Var_optWithFixStep}\n")
        file.write(f"Processing Time (Line Search Step): {time_fixed_step:.2f} seconds\n")
        file.write(f"Integral of the source: {integralS_x:.5e}\n")
        file.write(f"Error between Initial and Target Temperature: {costFunction(T_ini):.5f}\n")
        file.write(f"Error between Fixed step Optimization and Target Temperature: {error_fixed_step:.5f}\n")

    print(f"Résultats sauvegardés dans le fichier : {file_name}")



def solveProjectedGradientMethodWithStepOptimal(Var_ini, iterations, K_ref, dim_opt, T_ini, method_name="projected"):

    # Définir le nom du répertoire pour cette méthode
    dir_name = f"graphs/{method_name}_method/optimal Step/{iterations}_iterations"
    os.makedirs(dir_name, exist_ok=True)  # Crée le répertoire si nécessaire

    # Lancer le chronomètre pour mesurer le temps de traitement
    start_time = time.time()

    # Optimisation avec recherche de ligne
    Var_optWithLineSearch, errors_line_search = gd.projectedGradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt)
    TOptimalStep = simulator(0, Var_optWithLineSearch, dim_opt)
    print("Optimized design variables with line search step: ", Var_optWithLineSearch)

    # Vérifier que `errors_line_search` a bien la bonne taille
    if len(errors_line_search) != iterations:
        print(f"Erreur : `errors_line_search` a une longueur de {len(errors_line_search)}, mais {iterations} itérations étaient attendues.")
        return

    # Enregistrer les informations de la variable optimisée
    S_check = compute_source(Var_optWithLineSearch, dim_opt)
    integralS_x = np.sum(S_check) * h
    error_optimal_step = costFunction(TOptimalStep)

    # Mesurer le temps de traitement
    time_optimal_step = time.time() - start_time

    # Tracé log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_line_search, marker='o', label='Line Search Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.savefig(f"{dir_name}/error_{method_name}.png")

    # Tracer la température
    plt.figure(figsize=(12, 6))
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TOptimalStep, label='Line Search Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Initial, Target and Optimized Temperature')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{dir_name}/temp_{method_name}.png")

    # Tracer le terme source
    plt.figure(figsize=(10, 6))
    plt.plot(Xg, S_check, marker='o')
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.grid()
    plt.savefig(f"{dir_name}/source_{method_name}.png")

    # Créer le chemin complet pour le fichier .res
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\results"
    file_name = os.path.join(results_dir, f"{method_name}_method", "optimised_step", f"{iterations}_iterations",
                             "ProjectedMethod.res")

    # Créer les répertoires parents si ils n'existent pas
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as file:
        file.write(f"Optimized design variables with line search step: {Var_optWithLineSearch}\n")
        file.write(f"Processing Time (Line Search Step): {time_optimal_step:.2f} seconds\n")
        file.write(f"Integral of the source: {integralS_x:.5e}\n")
        file.write(f"Error between Initial and Target Temperature: {costFunction(T_ini):.5f}\n")
        file.write(f"Error between Line Search Optimization and Target Temperature: {error_optimal_step:.5f}\n")
        file.write(f"Total Processing Time: {time_optimal_step:.2f} seconds\n")

    print(f"Résultats sauvegardés dans le fichier : {file_name}")

















