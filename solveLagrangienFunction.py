from computeSource import *
from costFunction import *
from performGradient import *
from tqdm import tqdm  # Pour la barre de progression
from heatedBar import *
import matplotlib
matplotlib.use('Agg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé
import matplotlib.pyplot as plt

import numpy as np
from variables import *
from scipy.optimize import minimize_scalar, minimize
import itertools


def lagrangianFunction(T_ini, var_opt, dim_opt, beta):
    S = compute_source(var_opt, dim_opt)
    integraleofS = np.sum(S) * h
    return costFunction(T_ini) + beta * integraleofS

def lagrangianCompute_gradientByVarOpt(lambda_adj, Var_opt, dim_opt, beta):
    """
    Compute the gradient of the cost function with respect to Var_opt, including the energy constraint term.
    """
    S = compute_source(Var_opt, dim_opt)  # Heat source
    # Energy constraint gradient (additional term)
    energy_grad = np.sum(S) * h  # The energy constraint term
    # Original gradient: using the adjoint solution
    grad = np.zeros_like(Var_opt)
    integraleB_i = np.zeros_like(Var_opt)

    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            grad[i] += - lambda_adj[j] * math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h

    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            integraleB_i[i] +=  math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h

    energy_penalty_grad = beta * integraleB_i # Derivative of energy penalty term
    # Adjust the gradient
    grad += energy_penalty_grad

    return grad

def lagragianGradient_descent(Var_opt, alpha_1, alpha_2, iterations, K_ref, dim_opt, beta):
    """
    Perform gradient descent to minimize the cost function
    :param Var_opt: the initial design variables
    :param alpha: the learning rate
    :param iterations: number of iterations for the gradient descent
    :return: the optimized design variables
    """
    errors = []  # Liste pour stocker les erreurs à chaque itération
    for iter in tqdm(range(iterations), desc="Fixed Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)

        lambda_adj = compute_adjoint(T, T_star, K_ref)
        grad = lagrangianCompute_gradientByVarOpt(lambda_adj, Var_opt, dim_opt, beta)

        S = compute_source(Var_opt, dim_opt)
        integraleofS = np.sum(S) * h

        Var_opt -= alpha_1 * grad
        beta += alpha_2 * integraleofS

        cost = costFunction(T)
        #cost = lagrangianFunction(T, Var_opt, dim_opt, beta)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}")
    return Var_opt, errors


def lagrangianLine_search(u, grad_u, grad_beta, beta, dim_opt):
    """
    Effectue une recherche de ligne pour trouver les pas optimaux alpha_1 et alpha_2.
    """
    # Objectif pour alpha_1 (minimisation en fonction de u)
    def objective_1(alpha_1):
        u_new = u - alpha_1 * grad_u
        T = simulator(0, u_new, dim_opt)  # Simule le problème direct avec u modifié
        return lagrangianFunction(T, u_new, dim_opt, beta)

    # Objectif pour alpha_2 (mise à jour de beta)
    def objective_2(alpha_2):
        beta_new = beta + alpha_2 * grad_beta
        T = simulator(0, u, dim_opt)  # Simule sans changer u, car on optimise beta
        return -lagrangianFunction(T, u, dim_opt, beta_new)

    # Utilisation de `minimize_scalar` pour trouver les meilleurs alpha_1 et alpha_2
    result_1 = minimize_scalar(objective_1)
    result_2 = minimize_scalar(objective_2)
    return result_1.x, result_2.x


def lagrangianGradient_descent_with_line_search(Var_opt, iterations, K_ref, dim_opt, beta):
    """
    Perform gradient descent with optimal step size determined by line search
    :param Var_opt: the initial design variables
    :param iterations: number of iterations for the gradient descent
    :return: the optimized design variables
    """
    errors = []  # Liste pour stocker les erreurs à chaque itération
    for iter in tqdm(range(iterations), desc="Optimal Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)
        S = compute_source(Var_opt, dim_opt)
        integraleofS = np.sum(S) * h
        lambda_adj = compute_adjoint(T, T_star, K_ref)
        grad = compute_gradient(lambda_adj, Var_opt, dim_opt)
        #alpha = line_search(Var_opt, grad, costFunction, dim_opt)
        alpha_1, alpha_2 = lagrangianLine_search(Var_opt, grad, integraleofS, beta, dim_opt)
        Var_opt -= alpha * grad
        beta += alpha_2 * integraleofS
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors


def lagrangianGradient_descent_with_line_search_VGpt(Var_opt, iterations, K_ref, dim_opt, beta):
    errors = []
    for iter in tqdm(range(iterations), desc="Optimal Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)
        S = compute_source(Var_opt, dim_opt)
        integraleofS = np.sum(S) * h
        lambda_adj = compute_adjoint(T, T_star, K_ref)
        grad_u = lagrangianCompute_gradientByVarOpt(lambda_adj, Var_opt, dim_opt, beta)

        def line_search_objective(alpha):
            alpha_1, alpha_2 = alpha
            u_new = Var_opt - alpha_1 * grad_u
            beta_new = beta + alpha_2 * integraleofS
            T_new = simulator(0, u_new, dim_opt)
            return lagrangianFunction(T_new, u_new, dim_opt, beta_new)

        result = minimize(line_search_objective, [0.01, 0.01], bounds=[(0, None), (0, None)])
        alpha_1, alpha_2 = result.x

        Var_opt -= alpha_1 * grad_u
        beta += alpha_2 * integraleofS
        cost = costFunction(T)
        errors.append(cost)

    return Var_opt, errors


def lagrangien_difference_gradient_centered(Var_opt, epsilon, dim_opt, beta):
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
        error_perturbed = lagrangianFunction(T_perturbed, Var_perturbed, dim_opt, beta)

        Var_perturbed[i] -= 2 * epsilon
        T_perturbed = simulator(0, Var_perturbed, dim_opt)
        error_perturbed -= lagrangianFunction(T_perturbed, Var_perturbed, dim_opt, beta)

        grad[i] = error_perturbed / (2 * epsilon)

    return grad

def lagrangianGradient_descent_adaptive_step(Var_opt, iterations, K_ref, dim_opt, beta, alpha_1_init=0.1, alpha_2_init=0.01,
                                             reduction_factor_1=0.5, increase_factor_1=1.1,
                                             reduction_factor_2=0.7, increase_factor_2=1.05):
    errors = []
    alpha_1, alpha_2 = alpha_1_init, alpha_2_init

    for iter in tqdm(range(iterations), desc="Adaptive Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)
        S = compute_source(Var_opt, dim_opt)
        integraleofS = np.sum(S) * h
        lambda_adj = compute_adjoint(T, T_star, K_ref)
        grad_u = lagrangianCompute_gradientByVarOpt(lambda_adj, Var_opt, dim_opt, beta)

        # Calcul des nouvelles variables avec les pas adaptatifs
        u_new = Var_opt - alpha_1 * grad_u
        beta_new = beta + alpha_2 * integraleofS
        T_new = simulator(0, u_new, dim_opt)

        # Calcul de la nouvelle fonction de coût
        lagrangian_current = lagrangianFunction(T, Var_opt, dim_opt, beta)
        lagrangian_new = lagrangianFunction(T_new, u_new, dim_opt, beta_new)

        # Ajustement adaptatif des pas
        if lagrangian_new < lagrangian_current:
            alpha_1 *= increase_factor_1
            alpha_2 *= increase_factor_2
        else:
            alpha_1 *= reduction_factor_1
            alpha_2 *= reduction_factor_2

        # Mise à jour des variables d'optimisation
        Var_opt = u_new
        beta = beta_new
        cost = costFunction(T)
        errors.append(cost)

        #print(f"Iteration {iter+1}, Cost: {cost}, Alpha_1: {alpha_1}, Alpha_2: {alpha_2}")

    return Var_opt, errors


def comparePerformedGradient(dim_opt, K_ref, beta):
    # Compute the gradient with the two methods
    T = simulator(0, Var_ini, dim_opt)
    lambda_adj = compute_adjoint(T, T_star, K_ref)
    grad = lagrangianCompute_gradientByVarOpt(lambda_adj, Var_ini, dim_opt, beta)
    #grad = augmentedGradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt, pen)
    grad_fd = lagrangien_difference_gradient_centered(Var_ini, 1e-6, dim_opt, beta)

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

if __name__ == '__main__':

    #compare gradient method
    K_ref = FiniteElement.compute_conduction(0)

    # solve the problem for initial conditions
    T_ini = FiniteElement.simulator(0, Var_ini, dim_opt)

    iterations = 1000
    alpha_1 = 20
    alpha_2 = 0.01
    dim_opt = 6
    # Var_ini = np.full(dim_opt, 0.0)
    # Var_ini = np.array([-21.36202421, 34.90290328, -19.76296709, -32.44794981, 45.26660816, -21.05999423])
    # define var_ini random of size dim_opt
    Var_ini = np.random.rand(dim_opt)
    # pen=0.048
    beta = 0

    choice = 1

    if choice == 0:
        comparePerformedGradient(dim_opt, K_ref, beta)

    elif choice == 1:
        Var_optWithFixStep, errors_fixed_step = lagragianGradient_descent(Var_ini,
                                                                          alpha_1,
                                                                          alpha_2,
                                                                          iterations,
                                                                          K_ref,
                                                                          dim_opt,
                                                                          beta)
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
        plt.savefig("Error With lagrangian.png")

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
        plt.savefig("temp With lagrangian.png")

        # Plot source term
        plt.figure(figsize=(10, 6))
        plt.plot(Xg, S_check, marker='o')
        plt.xlabel('x')
        plt.ylabel('Source Term')
        plt.title('Source Term along x')
        plt.grid()
        plt.savefig("source with lagrangian.png")


    elif choice == 2:
        # find best alpha fixing beta and making a grid of alpha
        alpha_values = np.linspace(0, 20, 100)
        errors = []
        for alpha_1 in alpha_values:
            Var_optWithFixStep, errors_fixed_step = lagragianGradient_descent(Var_ini,
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
        plt.show()

    elif choice == 3:
        # find best beta fixing alpha and making a grid of beta
        alpha_values = np.linspace(0, 0.01, 100)
        errors = []
        for alpha_2 in alpha_values:
            Var_optWithFixStep, errors_fixed_step = lagragianGradient_descent(Var_ini,
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
        plt.show()

    elif choice == 4:
        #use oprimal step
        #Var_optWithOptimalStep, errors_optimal_step = lagrangianGradient_descent_adaptive_step(Var_ini,
        # Var_optWithOptimalStep, errors_optimal_step = lagrangianGradient_descent_with_line_search(Var_ini,
        #                                                                                     iterations,
        #                                                                                         K_ref,
        #                                                                                         dim_opt,
        #                                                                                         beta)

        Var_optWithOptimalStep, errors_optimal_step = lagrangianGradient_descent_with_line_search_VGpt(Var_ini,
                                                                                            iterations,
                                                                                            K_ref,
                                                                                            dim_opt,
                                                                                            beta)

        TOptimalStep = simulator(0, Var_optWithOptimalStep, dim_opt)
        print("Optimized design variables with optimal step: ", Var_optWithOptimalStep)

        # check energy nulle
        S_check = compute_source(Var_optWithOptimalStep, dim_opt)
        integralS_x = np.sum(S_check) * h
        print("Integral of the source: ", integralS_x)

        print("Error between Initial and Target Temperature: ", costFunction(TOptimalStep))

        # Tracer le graphique log-log des erreurs
        plt.figure(figsize=(10, 6))
        plt.loglog(range(1, iterations + 1), errors_optimal_step, marker='o', label='Optimal Step Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Error as a function of Iterations')
        plt.grid()
        plt.show()

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(X, T_ini, label='Initial Temperature')
        plt.plot(X, T_star, 'r--', label='Target Temperature')
        plt.plot(X, TOptimalStep, label='Optimal Step Optimization')
        plt.xlabel('x')
        plt.ylabel('Temperature')
        plt.title('Comparison of Initial, Target and Optimized Temperature')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

        # Plot source term
        plt.figure(figsize=(10, 6))
        plt.plot(Xg, S_check, marker='o')
        plt.xlabel('x')
        plt.ylabel('Source Term')
        plt.title('Source Term along x')
        plt.grid()
        plt.show()

    elif choice == 5:

        # Définir les valeurs possibles pour chaque paramètre
        alpha_1_init_values = [0.01, 0.1, 1]
        alpha_2_init_values = [0.001, 0.01, 0.1]
        reduction_factor_1_values = [0.5, 0.7, 0.9]
        increase_factor_1_values = [1.1, 1.2, 1.5]
        reduction_factor_2_values = [0.5, 0.7, 0.9]
        increase_factor_2_values = [1.05, 1.1, 1.3]

        # Grille de paramètres
        param_grid = itertools.product(
            alpha_1_init_values,
            alpha_2_init_values,
            reduction_factor_1_values,
            increase_factor_1_values,
            reduction_factor_2_values,
            increase_factor_2_values
        )

        best_params = None
        best_final_cost = float("inf")
        best_errors = None

        # Boucle sur chaque combinaison de paramètres
        for (alpha_1_init, alpha_2_init, reduction_factor_1, increase_factor_1,
             reduction_factor_2, increase_factor_2) in param_grid:

            # Exécuter la descente de gradient adaptative avec les paramètres actuels
            Var_opt, errors = lagrangianGradient_descent_adaptive_step(
                Var_ini,  # Paramètre initial pour Var_opt
                iterations=100,  # Choisissez un nombre d'itérations adapté
                K_ref=K_ref,
                dim_opt=dim_opt,
                beta=beta,
                alpha_1_init=alpha_1_init,
                alpha_2_init=alpha_2_init,
                reduction_factor_1=reduction_factor_1,
                increase_factor_1=increase_factor_1,
                reduction_factor_2=reduction_factor_2,
                increase_factor_2=increase_factor_2
            )

            final_cost = errors[-1]  # Coût final après toutes les itérations
            print(f"Params: {alpha_1_init}, {alpha_2_init}, {reduction_factor_1}, {increase_factor_1}, "
                  f"{reduction_factor_2}, {increase_factor_2} => Final Cost: {final_cost}")

            # Comparer avec le meilleur coût actuel et mettre à jour si on trouve mieux
            if final_cost < best_final_cost:
                best_final_cost = final_cost
                best_params = (alpha_1_init, alpha_2_init, reduction_factor_1,
                               increase_factor_1, reduction_factor_2, increase_factor_2)
                best_errors = errors  # Conserver l'évolution des erreurs pour le meilleur ensemble de paramètres

        print(f"\nBest Parameters: {best_params}")
        print(f"Best Final Cost: {best_final_cost}")



