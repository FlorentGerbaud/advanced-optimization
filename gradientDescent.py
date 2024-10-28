#__________________________________ import modules _______________________________________________________

from projectionMethod import *
import performGradient as pg
from LagrangianMethod import lagrangianFunction
from scipy.optimize import minimize_scalar

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

######################################################################################################
####################################### Penalization Method ##########################################
######################################################################################################

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
        print("alpha", alpha)
        Var_opt -= alpha * grad
        cost = augmentedCostFunction(T, Var_opt, dim_opt, pen)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors

def augmentedLine_search(u, grad_u, cost_function, dim_opt, pen):
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
        return augmentedCostFunction(T, u, dim_opt, pen)  # Return the cost function value

    # Use minimize_scalar to find the best alpha
    result = minimize_scalar(objective)
    return result.x  # Return the optimal alpha


######################################################################################################
####################################### Projection Method ############################################
######################################################################################################

def gradient_descentProjected(Var_opt, alpha, iterations, K_ref, dim_opt):
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

        #S = constraintFunction(Var_opt, dim_opt)
        grad_constraint = constraintGradient(Var_opt, dim_opt)

        projected_grad = grad - np.dot(grad, grad_constraint) * grad_constraint

        Var_opt -= alpha * projected_grad
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}")
    return Var_opt, errors

def projectedGradient_descent_with_line_search(Var_opt, iterations, K_ref, dim_opt):
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

        grad_constraint = constraintGradient(Var_opt, dim_opt)

        projected_grad = grad - np.dot(grad, grad_constraint) * grad_constraint

        alpha = line_search(Var_opt, grad, costFunction, dim_opt)
        Var_opt -= alpha * projected_grad
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors

######################################################################################################
####################################### Lagrangian Method ############################################
######################################################################################################

def lagragianGradient_descent(Var_opt, alpha_1, alpha_2, iterations, K_ref, dim_opt, beta):
    """
    Perform gradient descent with fixed step size
    :param Var_opt: the design variables
    :param alpha_1: the step size for the gradient descent for the design variables
    :param alpha_2: the step size for the gradient descent for the Lagrangian coefficient
    :param iterations: the number of iterations
    :param K_ref: the reference temperature
    :param dim_opt: the number of design variables
    :param beta: the Lagrangian coefficient
    :return: the optimized design variables and the errors
    """
    errors = []  # Liste pour stocker les erreurs à chaque itération
    for iter in tqdm(range(iterations), desc="Fixed Step Optimization"):
        T = simulator(0, Var_opt, dim_opt)

        lambda_adj = pg.compute_adjoint(T, T_star, K_ref)
        grad = pg.lagrangianCompute_gradientByVarOpt(lambda_adj, Var_opt, dim_opt, beta)

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
        lambda_adj = pg.compute_adjoint(T, T_star, K_ref)
        grad = pg.compute_gradient(lambda_adj, Var_opt, dim_opt)
        #alpha = line_search(Var_opt, grad, costFunction, dim_opt)
        alpha_1, alpha_2 = lagrangianLine_search(Var_opt, grad, integraleofS, beta, dim_opt)
        Var_opt -= alpha * grad
        beta += alpha_2 * integraleofS
        cost = costFunction(T)
        errors.append(cost)  # Ajouter l'erreur à la liste
        #print(f"Iteration {iter+1}, Cost Function: {cost}, Optimal Alpha: {alpha}")
    return Var_opt, errors