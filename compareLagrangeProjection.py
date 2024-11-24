import os
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap

from LagrangianMethod import *
from projectionMethod import *
import FEM as FiniteElement
from variables import *

def compare_methods(Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, beta, alpha_1, alpha_2):
    """
    Applique les méthodes de Lagrangien et de Projection (pas fixe et optimal)
    et trace les résultats.
    """

    # Création du répertoire de sauvegarde
    output_dir = f"graphs/comparisonLagrangeProjection_{iterations}"
    os.makedirs(output_dir, exist_ok=True)

    # Méthode du Lagrangien
    T_lagrangian, S_lagrangian, errors_lagrangian = solveLagrangienFunction(
        Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, beta, alpha_1, alpha_2
    )
    Var_ini = np.full(dim_opt, 0.0)
    # Méthode de projection avec pas fixe
    T_proj_fixed, S_proj_fixed, errors_proj_fixed = solveProjectedGradientMethodWithStepFixed(
        Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, method_name="projected"
    )
    Var_ini = np.full(dim_opt, 0.0)
    # Méthode de projection avec recherche de pas optimal
    T_proj_optimal, S_proj_optimal, errors_proj_optimal = solveProjectedGradientMethodWithStepOptimal(
        Var_ini, iterations, K_ref, dim_opt, T_ini, method_name="projected"
    )

    # Affichage des températures
    plt.figure(figsize=(10, 6))
    plt.plot(T_ini, label="Température initiale", color="blue")
    plt.plot(T_star, label="Température cible", linestyle="--", color="red")
    plt.plot(T_lagrangian, label="Méthode Lagrangien", color="green")
    plt.plot(T_proj_fixed, label="Projection avec pas fixe", color="orange")
    plt.plot(T_proj_optimal, label="Projection avec pas optimal",color="purple")
    plt.xlabel("Position (x)")
    plt.ylabel("Température")
    plt.title("Comparaison des températures")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "temperature_comparison.png"))
    plt.show()

    # Affichage des erreurs (log-log)
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(errors_lagrangian) + 1), errors_lagrangian, label="Méthode Lagrangien", color="green")
    plt.loglog(range(1, len(errors_proj_fixed) + 1), errors_proj_fixed, label="Projection avec pas fixe", color="orange")
    plt.loglog(range(1, len(errors_proj_optimal) + 1), errors_proj_optimal, label="Projection avec pas optimal", color="purple")
    plt.xlabel("Itérations")
    plt.ylabel("Erreur")
    plt.title("Évolution des erreurs (log-log)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "error_comparison.png"))
    plt.show()

    # Affichage des termes sources
    plt.figure(figsize=(10, 6))
    plt.plot(S_lagrangian, label="Méthode Lagrangien", color="green")
    plt.plot(S_proj_fixed, label="Projection avec pas fixe", color="orange")
    plt.plot(S_proj_optimal, label="Projection avec pas optimal", color="purple")
    plt.xlabel("Position (x)")
    plt.ylabel("Terme source")
    plt.title("Comparaison des termes sources")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "source_comparison.png"))
    plt.show()


if __name__ == '__main__':
    # Without noise
    K_ref = FiniteElement.compute_conduction(0)
    # Solve the problem for initial conditions
    T_ini = FiniteElement.simulator(0, Var_ini, dim_opt)

    compare_methods(Var_ini, alpha, iterations, K_ref, dim_opt, T_ini, beta, alpha_1, alpha_2)
