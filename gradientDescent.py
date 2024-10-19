#__________________________________ import modules _______________________________________________________

from performGradient import *
from tqdm import tqdm  # Pour la barre de progression
from scipy.optimize import minimize_scalar
from costFunction import *
from FEM import *




############################################################################################################
############################################ gradient_descent ##############################################
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

def augmentedGradient_descent(Var_opt, alpha, iterations, K_ref, dim_opt, pen):
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
        grad = augmentedCompute_gradient(lambda_adj, Var_opt, dim_opt, pen)
        Var_opt -= alpha * grad
        cost = augmentedCostFunction(T, Var_opt, dim_opt, pen)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}")
    return Var_opt, errors

def augmentedGradient_descent_with_line_search(Var_opt, iterations, K_ref, dim_opt, pen):
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
        grad = augmentedCompute_gradient(lambda_adj, Var_opt, dim_opt, pen)
        alpha = augmentedLine_search(Var_opt, grad, costFunction, dim_opt, pen)
        Var_opt -= alpha * grad
        cost = augmentedCostFunction(T, Var_opt, dim_opt, pen)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors

def augmentedLine_search(u, grad_u, Var_opt, dim_opt, pen):
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
        return augmentedCostFunction(T, Var_opt, dim_opt, pen)  # Return the cost function value

    # Use minimize_scalar to find the best alpha
    result = minimize_scalar(objective)
    return result.x  # Return the optimal alpha