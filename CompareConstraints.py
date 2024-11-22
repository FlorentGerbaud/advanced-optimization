import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from costFunction import *
import gradientDescent as gd
from projectionMethod import *



# Fonction pour résoudre avec pas fixe et enregistrer les résultats
def solve_and_plot(alpha, iterations, K_ref, dim_opt, method_name="projected"):
    # Créer les répertoires pour les graphiques
    dir_name = f"graphs/{method_name}_method/fixed Step/{iterations}_iterations"
    os.makedirs(dir_name, exist_ok=True)

    # Listes pour stocker les résultats
    errors_all = []
    temps_all = []
    sources_all = []

    # Faire varier Var_ini de [0,0,0,0,0,0] à [5,5,5,5,5,5] et optimiser
    for i in range(6):
        Var_iniLoc = np.full(dim_opt, float(i))  # Variable d'entrée
        # Résoudre le problème pour les conditions initiales
        T_ini = FiniteElement.simulator(0, Var_iniLoc, dim_opt)
        TFixedStep, S_check, errors_fixed_step = solveProjectedGradientMethodWithStepFixed(Var_iniLoc, alpha,
                                                                                           iterations,
                                                                                           K_ref, dim_opt, T_ini,
                                                                                           method_name)

        # Ajouter les résultats
        errors_all.append(errors_fixed_step)
        temps_all.append(TFixedStep)
        sources_all.append(S_check)

    # Tracer et sauvegarder les graphiques

    # 1. Graphique des erreurs (log-log)
    plt.figure(figsize=(10, 6))
    for i, errors in enumerate(errors_all):
        plt.loglog(range(1, iterations + 1), errors, marker='o', label=f'Var_ini = {i}')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.legend()
    plt.savefig(f"{dir_name}/error_loglog.png")

    # 2. Graphique des températures
    plt.figure(figsize=(12, 6))
    for i, temps in enumerate(temps_all):
        plt.plot(X, temps, label=f'Var_ini = {i}')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Optimized Temperatures')
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/temperature_curves.png")

    # 3. Graphique des termes sources avec aire sous la courbe
    plt.figure(figsize=(10, 6))
    for i, sources in enumerate(sources_all):
        # Calculer l'aire sous la courbe des sources
        area = np.sum(sources) * h
        # Arrondir l'aire à 1 chiffre significatif
        area_rounded = round(area, 1)

        # Tracer la courbe avec l'aire affichée comme label
        plt.plot(Xg, sources, marker='o', label=f'Var_ini = {i} (Aire = {area_rounded})')

    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/source_terms.png")

    # Sauvegarder les résultats dans un fichier
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\results"
    file_name = os.path.join(results_dir, f"{method_name}_method", "fixed_step", f"{iterations}_iterations",
                             "ProjectedMethod_results.res")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as file:
        file.write(f"Results for projected gradient method with fixed step\n")
        for i, errors in enumerate(errors_all):
            file.write(f"Var_ini = {i}, Final Error = {errors[-1]:.5e}\n")

    print(f"Results saved in: {file_name}")


# Fonction pour résoudre avec pas optimal et enregistrer les résultats
def solve_and_plot_with_optimal_step(iterations, K_ref, dim_opt, method_name="projected"):
    # Créer les répertoires pour les graphiques
    dir_name = f"graphs/{method_name}_method/optimal Step/{iterations}_iterations"
    os.makedirs(dir_name, exist_ok=True)

    # Listes pour stocker les résultats
    errors_all = []
    temps_all = []
    sources_all = []

    # Faire varier Var_ini de [0,0,0,0,0,0] à [5,5,5,5,5,5] et optimiser
    for i in range(6):
        Var_iniLoc = np.full(dim_opt, float(i))  # Variable d'entrée
        # Résoudre le problème pour les conditions initiales
        T_ini = FiniteElement.simulator(0, Var_iniLoc, dim_opt)
        TOptimalStep, S_check, errors_optimal_step = solveProjectedGradientMethodWithStepOptimal(Var_iniLoc, iterations,
                                                                                                 K_ref, dim_opt, T_ini,
                                                                                                 method_name)

        # Ajouter les résultats
        errors_all.append(errors_optimal_step)
        temps_all.append(TOptimalStep)
        sources_all.append(S_check)

    # Tracer et sauvegarder les graphiques

    # 1. Graphique des erreurs (log-log)
    plt.figure(figsize=(10, 6))
    for i, errors in enumerate(errors_all):
        plt.loglog(range(1, iterations + 1), errors, marker='o', label=f'Var_ini = {i}')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error as a function of Iterations')
    plt.grid()
    plt.legend()
    plt.savefig(f"{dir_name}/error_loglog.png")

    # 2. Graphique des températures
    plt.figure(figsize=(12, 6))
    for i, temps in enumerate(temps_all):
        plt.plot(X, temps, label=f'Var_ini = {i}')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Comparison of Optimized Temperatures')
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/temperature_curves.png")

    # 3. Graphique des termes sources avec aire sous la courbe
    plt.figure(figsize=(10, 6))
    for i, sources in enumerate(sources_all):
        # Calculer l'aire sous la courbe des sources
        area = np.sum(sources) * h
        # Arrondir l'aire à 1 chiffre significatif
        area_rounded = round(area, 1)

        # Tracer la courbe avec l'aire affichée comme label
        plt.plot(Xg, sources, marker='o', label=f'Var_ini = {i} (Aire = {area_rounded})')

    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.title('Source Term along x')
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/source_terms.png")

    # Sauvegarder les résultats dans un fichier
    results_dir = r"C:\Users\flore\PycharmProjects\advanced-optimization\results"
    file_name = os.path.join(results_dir, f"{method_name}_method", "optimal_step", f"{iterations}_iterations",
                             "ProjectedMethod_results.res")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as file:
        file.write(f"Results for projected gradient method with optimal step\n")
        for i, errors in enumerate(errors_all):
            file.write(f"Var_ini = {i}, Final Error = {errors[-1]:.5e}\n")

    print(f"Results saved in: {file_name}")


if __name__ == '__main__':


    # without noise
    K_ref = FiniteElement.compute_conduction(0)
    #solve_and_plot(alpha, iterations, K_ref, dim_opt)
    solve_and_plot_with_optimal_step(iterations, K_ref, dim_opt)