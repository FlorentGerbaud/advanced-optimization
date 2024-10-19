from gradientDescent import *
# Run gradient descent with fixed learning rate
from variables import *
import FEM as FiniteElement
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np

choice = 0

# without noise
K_ref = FiniteElement.compute_conduction(0)

# solve the problem for initial conditions
T_ini = FiniteElement.simulator(0, Var_ini, dim_opt)

iterations = 500
alpha = 20
dim_opt = 6
Var_ini = np.full(dim_opt, 0.0)
pen=0.048

if choice == 0:

    Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent(Var_ini,
                                                                        alpha,
                                                                        iterations,
                                                                        K_ref,
                                                                        dim_opt,
                                                                        pen)
    TFixedStep = simulator(0, Var_optWithFixStep, dim_opt)
    print("Optimized design variables with fixed step: ", Var_optWithFixStep)

    # check energy nulle
    S_check = compute_source(Var_optWithFixStep, dim_opt)
    integralS_x = np.sum(S_check) * h
    print("Integral of the source: ", integralS_x)

    print("Error between Initial and Target Temperature: ", costFunction(TFixedStep))
    print("Error between Initial and Target Temperature with penalty term: ", augmentedCostFunction(TFixedStep, Var_optWithFixStep, dim_opt, pen))

    # Tracer le graphique log-log des erreurs
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, iterations + 1), errors_fixed_step, marker='o', label='Fixed Step Error')
    plt.xlabel('Number of Iterations (log scale)')
    plt.ylabel('Error (log scale)')
    plt.title('Comparison of Errors in Log-Log Scale')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    plt.show()

    # Plot source term
    plt.figure(figsize=(12, 6))  # Agrandir le graphique
    plt.plot(Xg, S_check, label='Source Term', color='blue')  # Tracer la source avec Xg
    plt.xlabel('x')
    plt.ylabel('Source Term')
    plt.legend(loc="lower right")
    plt.title(
        f'Source Term After Optimization (Iterations: {iterations})')  # Inclure le nombre d'itérations
    plt.grid()
    plt.show()

else:
    # Define the penalty values to test
    penalty_values = np.linspace(0.04, 0.05, 10)
    errors = []
    integralValues = []

    # Iterate over penalty values to compute errors and integral values
    for pen in penalty_values:
        Var_optWithFixStep, errors_fixed_step = augmentedGradient_descent(Var_ini,
                                                                          alpha,
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
    plt.loglog(penalty_values, errors, label="Error", marker='o')
    plt.xlabel("Penalty Value")
    plt.ylabel("Error")
    plt.title("Error vs Penalty Value (Log-Log Scale)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting Integral of S in Log-Log Scale
    plt.figure(figsize=(10, 5))
    plt.loglog(penalty_values, integralValues, label="Integral of S", marker='x')
    plt.xlabel("Penalty Value")
    plt.ylabel("Integral of S")
    plt.title("Integral of S vs Penalty Value (Log-Log Scale)")
    plt.grid(True)
    plt.legend()
    plt.show()
