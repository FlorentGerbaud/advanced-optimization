#_______________________________________ import modules _______________________________________#

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sympy.physics.control.control_plots import matplotlib
from tqdm import tqdm  # Pour la barre de progression
from scipy.optimize import minimize_scalar
import time
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' si vous avez PyQt5 installé

#_______________________________________ define variables _______________________________________#

# domain length
l = 1
# number of elements
n_el = 200

# element size
h = l / n_el
# number of nodes
n_no = n_el + 1
# nodes coordinates
X = np.linspace(0, 1, n_no)
# integration points coordinates
Xg = 0.5 * (X[:-1] + X[1:])

# target temperature
T_star = np.ones(X.shape) + (X / l) ** 2

# Parameters for optimization
alpha = 20
iterations = 200

# initial design vector that parameterizes heat sources
dim_opt = 50
Var_ini = np.full(dim_opt, 0.0)
#with optimal descent gradient
#Var_ini = np.array([-21.36202421, 34.90290328, -19.76296709, -32.44794981, 45.26660816, -21.05999423])
rng = np.random.default_rng(seed=2024)

#_______________________________________ define functions _______________________________________#

############################################################################################################
############################################ Compute_conduction ############################################
############################################################################################################

# function that compute the conduction coefficient with possible noise

def compute_conduction(noise_val):
    """
    Compute the conduction coefficient with possible noise
    :param noise_val: the noise value
    :return: the conduction coefficient
    """
    noise = rng.random(Xg.shape) * noise_val
    K = np.ones(Xg.shape) + 0.4 * np.sin(4 * np.pi * Xg / l) + 0.3 * np.sin(12 * np.pi * Xg / l) + noise
    return K

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

############################################################################################################
############################################ compute_matrix ################################################
############################################################################################################

def compute_matrix(K):
    """
    Compute the matrix of the problem
    :param K: the conduction coefficient
    :return: the matrix of the problem
    """
    # Finite-Element matrix initialization
    M = np.zeros((n_no, n_no))

    # Boundary conditions
    M[0][0] = 1
    M[n_no - 1][n_no - 1] = 1

    # Internal coeff
    for i in range(1, n_no - 1):
        M[i][i] = (K[i - 1] + K[i]) / h
        M[i][i - 1] = -K[i - 1] / h
        M[i][i + 1] = -K[i] / h

    return M

############################################################################################################
############################################ compute_rhs ###################################################
############################################################################################################

def compute_rhs(S):
    """
    Compute the right-hand side of the problem
    :param S: the heat source
    :return: the right-hand side of the problem
    """
    # Finite-Element right-hand side initialization
    Rhs = np.zeros((n_no, 1))

    # Boundary conditions
    Rhs[0] = 1
    Rhs[n_no - 1] = 2

    # internal coeff
    for i in range(1, n_no - 1):
        Rhs[i] = (S[i - 1] + S[i]) * h / 2

    return Rhs

############################################################################################################
############################################ simulator #####################################################
############################################################################################################

def simulator(noise, Var, dim_opt):
    """
    Simulate the problem
    :param noise: The noise value
    :param Var: the vector of coefficients
    :return: the temperature
    """
    # conduction
    K = compute_conduction(noise)

    # matrix
    M = compute_matrix(K)

    # compute heat source
    Src = compute_source(Var, dim_opt)

    # right-hand side
    Rhs = compute_rhs(Src)

    # Finite-element solution
    T = np.matmul(np.linalg.inv(M), Rhs).reshape((n_no))

    return T

############################################################################################################
############################################ costFunction #################################################
############################################################################################################

def costFunction(T_ini):
    """
    Compute the cost function
    :return: the norm of the difference between the initial temperature and the target temperature
    """
    return 0.5 * sum((T_ini - T_star) ** 2) * h

############################################################################################################
############################################ compute_adjoint ################################################
############################################################################################################

def compute_adjoint(T, T_star, K):
    """
    Solve the adjoint equation to compute the Lagrange multiplier lambda
    :param T: the computed temperature
    :param T_star: the target temperature
    :param K: the conduction coefficient
    :return: the Lagrange multiplier lambda
    """
    # Finite-Element matrix initialization
    M_adj = np.zeros((n_no, n_no))
    Rhs_adj = np.zeros((n_no, 1))

    # Boundary conditions for the adjoint problem (Dirichlet boundary conditions, lambda = 0 at boundaries)
    M_adj[0][0] = 1
    M_adj[n_no - 1][n_no - 1] = 1
    # boundary conditions allwos to have lambda = 0 at boundaries
    # so l(x).k(x).grad(T) = 0 at boundaries
    #and d/dx(l(x).k(x).T) = 0 at boundaries
    Rhs_adj[0] = 0
    Rhs_adj[n_no - 1] = 0

    # Internal coefficients
    for i in range(1, n_no - 1):
        M_adj[i][i] = (K[i - 1] + K[i]) / h
        M_adj[i][i - 1] = -K[i - 1] / h
        M_adj[i][i + 1] = -K[i] / h
        #Rhs_adj[i] = 2 * (T[i] - T_star[i])  # RHS of the adjoint equation
        Rhs_adj[i] = - (T[i] - T_star[i]) * h # RHS of the adjoint equation

    # Solve the adjoint equation
    lambda_adj = np.matmul(np.linalg.inv(M_adj), Rhs_adj).reshape((n_no))

    return lambda_adj

############################################################################################################
############################################ compute_gradient ################################################
############################################################################################################

def compute_gradient(lambda_adj, Var_opt, dim_opt):
    """
    Compute the gradient of the cost function with respect to the design variables Var_opt
    :param lambda_adj: the Lagrange multiplier
    :param Var_opt: the vector of design variables
    :return: the gradient of the cost function
    """
    grad = np.zeros(dim_opt)

    for i in range(dim_opt):
        for j in range(Xg.shape[0]):
            grad[i] += - lambda_adj[j] * math.comb(dim_opt - 1, i) * Xg[j] ** i * (1 - Xg[j]) ** (dim_opt - 1 - i) * h

    return grad

def finite_difference_gradient_centered(Var_opt, epsilon, dim_opt):
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
        error_perturbed = costFunction(T_perturbed)

        Var_perturbed[i] -= 2 * epsilon
        T_perturbed = simulator(0, Var_perturbed, dim_opt)
        error_perturbed -= costFunction(T_perturbed)

        grad[i] = error_perturbed / (2 * epsilon)

    return grad

############################################################################################################
############################################ gradient_descent ################################################
############################################################################################################

def gradient_descent(Var_opt, alpha, iterations, K_ref, dim_opt):
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
        grad = compute_gradient(lambda_adj, Var_opt, dim_opt)
        Var_opt -= alpha * grad
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}")
    return Var_opt, errors


def line_search(u, grad_u, cost_func, dim_opt):
    """
    Perform line search to find the optimal step size alpha.
    :param u: the current design variables
    :param grad_u: the gradient at the current step
    :param cost_func: the cost function
    :return: optimal step size alpha
    """
    # Define the function to minimize w.r.t alpha
    def objective(alpha):
        # Evaluate the cost function at u - alpha * grad_u
        # we want to minimize the cost function to find the better alpha that find the better soltion for u
        # so we perform u_new and then we perform the forward problem simulation
        u_new = u - alpha * grad_u
        # we perform T because the cost function is defined as a function of T and not u
        T = simulator(0, u_new, dim_opt)  # Forward problem simulation
        return costFunction(T)  # Return the cost function value

    # Use minimize_scalar to find the best alpha
    result = minimize_scalar(objective)
    return result.x  # Return the optimal alpha


def gradient_descent_with_line_search(Var_opt, iterations, K_ref, dim_opt):
    """
    Perform gradient descent with optimal step size determined by line search
    :param Var_opt: the initial design variables
    :param iterations: number of iterations for the gradient descent
    :return: the optimized design variables
    """
    errors = []  # Liste pour stocker les erreurs à chaque itération
    for iter in tqdm(range(iterations), desc="Optimal Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)
        lambda_adj = compute_adjoint(T, T_star, K_ref)
        grad = compute_gradient(lambda_adj, Var_opt, dim_opt)
        alpha = line_search(Var_opt, grad, costFunction, dim_opt)
        Var_opt -= alpha * grad
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors


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

def comparePerformedGradient(dim_opt):
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




#_______________________________________ main program _______________________________________#
##############################################################################################

if __name__ == '__main__':

    # without noise
    K_ref = compute_conduction(0)

    # solve the problem for initial conditions
    T_ini = simulator(0, Var_ini, dim_opt)

    # choice = 1 for the comparison of the two methods
    # choice = 2 compare gradient value for the two methods
    # choice = 3 compare fixed step with different alpha valuees from 0.01 to 20
    # choice = 4 compare the error with the target for both method from fixed step and optimal step with number of Var from 2 to 100

    choice = 4

    if choice == 1:

        optimize_problem(dim_opt,
                         K_ref,
                         T_ini,
                         l,
                         n_el,
                         noise=0,
                         alpha=20,
                         iterations=1000,
                         results_filename="results/",
                         graphs_output="graphs/")

    elif choice == 2:

        comparePerformedGradient(dim_opt)

    elif choice == 3:

        findThebestAllphaValues(K_ref, dim_opt)

    elif choice == 4:

        findBestNumberOfVal(K_ref, alpha, iterations)

