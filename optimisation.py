#______________________________________ import libraries __________________________________________________

import matplotlib.pyplot as plt
import time
import os
from gradientDescent import *


def optimize_problem(dim_opt, K_ref, T_ini, l=1, n_el=200, noise=0.0, alpha=0.9, iterations=200, results_filename="results/", graphs_output="graphs/"):
    iterations_filename = results_filename + "noise_" + str(noise) + "_iterations_" + str(iterations) + ".Results"
    os.makedirs(os.path.dirname(iterations_filename), exist_ok=True)
    graphs_output = graphs_output + "noise_" + str(noise) + "_iterations_" + str(iterations) + "/"
    os.makedirs(os.path.dirname(graphs_output), exist_ok=True)


    with open(iterations_filename, "w", encoding='utf-8') as f:

        # Mesurer le temps de traitement total
        start_time = time.time()

        # Run gradient descent with fixed learning rate
        start_fixed_step_time = time.time()  # Démarrer le chronomètre pour la méthode à pas fixe
        Var_optWithFixStep, errors_fixed_step = gradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt)
        TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
        end_fixed_step_time = time.time()  # Arrêter le chronomètre pour la méthode à pas fixe
        fixed_step_time = end_fixed_step_time - start_fixed_step_time

        output = f"Optimized design variables with fixed step: {Var_optWithFixStep}\n"
        print(output)
        f.write(output)

        output = f"Processing Time (Fixed Step): {fixed_step_time:.2f} seconds\n"
        print(output)
        f.write(output)

        # Run gradient descent with line search
        start_optimal_step_time = time.time()  # Démarrer le chronomètre pour la méthode de recherche optimale
        Var_optWithOptimalStep, errors_optimal_step = gradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt)
        TOptimalStep = simulator(0, Var_optWithOptimalStep, dim_opt)
        end_optimal_step_time = time.time()  # Arrêter le chronomètre pour la méthode à pas fixe
        optimal_step_time = end_optimal_step_time - start_optimal_step_time

        output = f"Optimized design variables with optimal step: {Var_optWithOptimalStep}\n"
        print(output)
        f.write(output)

        output = f"Processing Time (Optimal Step): {optimal_step_time:.2f} seconds\n"
        print(output)
        f.write(output)

        # Calculate errors
        error_initial = costFunction(T_ini)
        error_fixed_step = costFunction(TFixedStep)
        error_optimal_step = costFunction(TOptimalStep)

        # Print errors
        output = f"Error between Initial and Target Temperature: {error_initial:.5f}\n"
        print(output)
        f.write(output)

        output = f"Error between Fixed Step Optimization and Target Temperature: {error_fixed_step:.5f}\n"
        print(output)
        f.write(output)

        output = f"Error between Optimal Step Optimization and Target Temperature: {error_optimal_step:.5f}\n"
        print(output)
        f.write(output)

        # Plot of the conduction coefficient along x
        plt.figure(figsize=(10, 5))  # Agrandir le graphique
        plt.plot(Xg, K_ref)
        plt.xlabel('x')
        plt.ylabel('Conduction Coefficient')
        plt.title('Conduction Coefficient along x')
        plt.grid()
        plt.savefig(f"{graphs_output}Conduction_Coefficient.png")  # Enregistrer le graphique
        plt.show()

        # Tracer le graphique log-log des erreurs
        plt.figure(figsize=(10, 6))
        plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
        plt.loglog(range(1, iterations + 1), errors_optimal_step, marker='s', label='Optimal Step Error')

        plt.xlabel('Number of Iterations (log scale)')
        plt.ylabel('Error (log scale)')
        plt.title('Comparison of Errors in Log-Log Scale')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{graphs_output}Errors_Comparison.png")  # Enregistrer le graphique
        plt.show()

        # Plot results
        plt.figure(figsize=(12, 6))  # Agrandir le graphique
        plt.plot(X, T_ini, label='Initial Temperature')
        plt.plot(X, T_star, 'r--', label='Target Temperature')
        plt.plot(X, TFixedStep, label='Fixed Step Optimization')
        plt.plot(X, TOptimalStep, label='Optimal Step Optimization')
        plt.xlabel('x')
        plt.ylabel('Temperature')
        plt.legend(loc="lower right")
        plt.title(
            f'Temperature Profiles After Optimization (Iterations: {iterations})')  # Inclure le nombre d'itérations
        plt.grid()

        # Annoter les erreurs sur le graphique
        plt.text(0.5, 0.5, f'Error (Initial): {error_initial:.5f}', fontsize=10, ha='center')
        plt.text(0.5, 0.45, f'Error (Fixed Step): {error_fixed_step:.5f}', fontsize=10, ha='center')
        plt.text(0.5, 0.4, f'Error (Optimal Step): {error_optimal_step:.5f}', fontsize=10, ha='center')

        plt.savefig(f"{graphs_output}Temperature_Profiles.png")  # Enregistrer le graphique
        plt.show()

        # Mesurer et imprimer le temps de traitement total
        end_time = time.time()
        processing_time = end_time - start_time
        output = f"Total Processing Time: {processing_time:.2f} seconds\n"
        print(output)
        f.write(output)

def comparePerformedGradient(dim_opt, K_ref):
    # Compute the gradient with the two methods
    T = simulator(0, Var_ini, dim_opt)
    lambda_adj = compute_adjoint(T, T_star, K_ref)
    grad = compute_gradient(lambda_adj, Var_ini, dim_opt)
    grad_fd = finite_difference_gradient_centered(Var_ini, 1e-6, dim_opt)

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


def findThebestAllphaValues(K_ref, dim_opt):
    # Perform fixed step optimization with different alpha values
    alphas = np.linspace(0.01, 20, 20)
    errors_fixed_step = []

    for alpha in alphas:
        Var_opt, errors = gradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt)
        errors_fixed_step.append(errors[-1])

    # Plot the errors for different alpha values
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, errors_fixed_step, marker='o')
    plt.xlabel('Learning Rate (Alpha)')
    plt.ylabel('Error')
    plt.title('Error vs Learning Rate')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("Error_vs_Learning_Rate.png")


def findBestNumberOfVal(K_ref, alpha, iterations):
    # Perform optimization with different number of design variables
    errorsInlineSearch = []
    errorsFixedStep = []
    design_vars = range(2, 101, 10)  # Range with steps of 10

    for dim_opt in design_vars:
        Var_ini = np.full(dim_opt, 0.0)  # Initialize design variables
        Var_opt, error = gradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt)
        errorsFixedStep.append(error[-1])  # Append the last error from the list
        #Var_opt, error = gradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt)
        Var_opt, error = gradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt)
        errorsInlineSearch.append(error[-1])  # Append the last error from the list

    #print("Errors for different number of design variables:", errors)

    # Plot the errors for different number of design variables on two separate graphs
    plt.figure(figsize=(10, 6))
    plt.plot(design_vars, errorsFixedStep, marker='o')
    plt.xlabel('Number of Design Variables')
    plt.ylabel('Error')
    plt.title('Error vs Number of Design Variables (Fixed Step)')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("Error_vs_Number_of_Design_Variables_Fixed_Step.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(design_vars, errorsInlineSearch, marker='o')
    plt.xlabel('Number of Design Variables')
    plt.ylabel('Error')
    plt.title('Error vs Number of Design Variables (Optimal Step)')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("Error_vs_Number_of_Design_Variables_Optimal_Step.png")
    plt.show()