from gradientDescent import *
# Run gradient descent with fixed learning rate
from variables import *
import FEM as FiniteElement
import matplotlib
matplotlib.use('Agg')  # or another interactive backend like 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np



def comparePerformedGradient(dim_opt, K_ref, pen):
    # Compute the gradient with the two methods
    T = simulator(0, Var_ini, dim_opt)
    lambda_adj = compute_adjoint(T, T_star, K_ref)
    grad = augmentedCompute_gradient(lambda_adj, Var_ini, dim_opt, pen)
    #grad = augmentedGradient_descent_with_line_search(Var_ini, iterations, K_ref, dim_opt, pen)
    grad_fd = augmentedFinite_difference_gradient_centered(Var_ini, 1e-6, dim_opt, pen)

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

choice = 0

# without noise
K_ref = FiniteElement.compute_conduction(0)

# solve the problem for initial conditions
T_ini = FiniteElement.simulator(0, Var_ini, dim_opt)

iterations = 500
alpha = 20
dim_opt = 6
#Var_ini = np.full(dim_opt, 0.0)
#Var_ini = np.array([-21.36202421, 34.90290328, -19.76296709, -32.44794981, 45.26660816, -21.05999423])
#define var_ini random of size dim_opt
Var_ini = np.random.rand(dim_opt)
#pen=0.048
pen_fixe = 0.0247
pen_opt = 0.0247
if choice == 0:

    # Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent_with_line_search(Var_ini,
    #                                                                     iterations,
    #                                                                     K_ref,
    #                                                                     dim_opt,
    #                                                                     pen_opt)
    Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent(Var_ini, alpha, iterations, K_ref, dim_opt, pen_fixe)
    TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
    print("Optimized design variables with fixed step: ", Var_optWithFixStep)

    # check energy nulle
    S_check = compute_source(Var_optWithFixStep, dim_opt)
    integralS_x = np.sum(S_check) * h
    print("Integral of the source: ", integralS_x)

    print("Error between Initial and Target Temperature: ", costFunction(TFixedStep))
    print("Error between Initial and Target Temperature with penalty term: ", augmentedCostFunction(TFixedStep, Var_optWithFixStep, dim_opt, pen_opt))

    # Tracer le graphique log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
    plt.xlabel('Number of Iterations (log scale)')
    plt.ylabel('Error (log scale)')
    plt.title('Comparison of Errors in Log-Log Scale')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error.png")

    # Plot results
    plt.figure(figsize=(12, 6))  # Agrandir le graphique
    plt.plot(X, T_ini, label='Initial Temperature')
    plt.plot(X, T_star, 'r--', label='Target Temperature')
    plt.plot(X, TFixedStep, label='Fixed Step Optimization')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.legend(loc="lower right")
    plt.title(
        f'Temperature Profiles After Optimization (Iterations: {iterations})')  # Inclure le nombre d'itérations
    plt.grid()
    plt.savefig("temperature.png")

    # Plot source term
    plt.figure(figsize=(12, 6))  # Agrandir le graphique
    plt.plot(Xg, S_check, label='Source Term', color='blue')  # Tracer la source avec Xg
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.legend(loc="lower right")
    plt.title(
        f'Source Term After Optimization (Iterations: {iterations})')  # Inclure le nombre d'itérations
    plt.grid()
    plt.savefig("source-term1.png")

elif choice == 1:
    # Define the penalty values to test
    penalty_values = np.linspace(10e-6, 10e-3, 10)
    errors = []
    integralValues = []

    # Iterate over penalty values to compute errors and integral values
    for pen in penalty_values:
        Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent_with_line_search(Var_ini,
                                                                          iterations,
                                                                          K_ref,
                                                                          dim_opt,
                                                                          pen)
        TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
        errors.append(costFunction(TFixedStep))  # Cost function
        S_check = compute_source(Var_optWithFixStep, dim_opt)
        integralS_x = np.sum(S_check) * h
        integralValues.append(integralS_x)  # Integral of S

    # Plotting Error in Log-Log Scale
    plt.figure(figsize=(10, 5))
    plt.plot(penalty_values, errors, label="Error", marker='o')
    plt.xlabel("Penalty Value")
    plt.ylabel("Error")
    plt.title("Error vs Penalty Value (Log-Log Scale)")
    plt.grid(True)
    plt.legend()
    plt.savefig("error-pen")

    # Plotting Integral of S in Log-Log Scale
    plt.figure(figsize=(10, 5))
    plt.plot(penalty_values, integralValues, label="Integral of S", marker='x')
    plt.xlabel("Penalty Value")
    plt.ylabel("Integral of S")
    plt.title("Integral of S vs Penalty Value (Log-Log Scale)")
    plt.grid(True)
    plt.legend()
    plt.savefig("integral-pen")

elif choice == 2:
    comparePerformedGradient(dim_opt, K_ref, pen_fixe)
