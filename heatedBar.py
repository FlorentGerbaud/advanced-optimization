#_______________________________________ import modules _______________________________________#

from variables import *
import costFunction as cf
import FEM as FiniteElement
import optimisation as opt
import projectionMethod as proj
import LagrangianMethod as lag
import CompareConstraints as cc

#_______________________________________ main program _______________________________________#
##############################################################################################

if __name__ == '__main__':

    # without noise
    K_ref = FiniteElement.compute_conduction(0)

    # solve the problem for initial conditions
    T_ini = FiniteElement.simulator(0, Var_ini, dim_opt)

    # choice = 1 for the comparison of the two methods
    # choice = 2 compare gradient value for the two methods
    # choice = 3 compare fixed step with different alpha valuees from 0.01 to 20
    # choice = 4 compare the error with the target for both method from fixed step and optimal step with number of Var from 2 to 100

    choice = 5

    if choice == 1:

        opt.optimize_problem(dim_opt,
                         K_ref,
                         T_ini,
                         l,
                         n_el,
                         noise=0,
                         alpha=20,
                         iterations=100,
                         results_filename="results/",
                         graphs_output="graphs/")

    elif choice == 2:

        opt.comparePerformedGradient(dim_opt, K_ref)

    elif choice == 3:

        opt.findThebestAllphaValues(K_ref, dim_opt)

    elif choice == 4:

        opt.findBestNumberOfVal(K_ref, alpha, iterations)

    elif choice == 5:

        proj.solveProjectedGradientMethodWithStepFixed(Var_ini = Var_ini
                                                       , alpha = alpha,
                                                       iterations = iterations,
                                                       K_ref = K_ref,
                                                       dim_opt = dim_opt,
                                                       T_ini = T_ini)

    elif choice == 6:

        proj.solveProjectedGradientMethodWithStepOptimal(Var_ini = Var_ini
                                                         , iterations = iterations,
                                                         K_ref = K_ref,
                                                         dim_opt = dim_opt,
                                                         T_ini = T_ini)


    elif choice == 7:

        lag.solveLagrangienFunction(Var_ini = Var_ini
                                      , alpha = alpha
                                      , iterations = iterations
                                      , K_ref = K_ref
                                      , dim_opt = dim_opt
                                      , T_ini = T_ini
                                      , beta = beta
                                      , alpha_1 = alpha_1
                                      , alpha_2 = alpha_2)

    elif choice == 8:

        lag.findBestAlpha_1(Var_ini = Var_ini,
                            alpha_1=alpha_1,
                            alpha_2=alpha_2,
                            iterations=iterations,
                            K_ref=K_ref,
                            dim_opt=dim_opt,
                            beta=beta)

    elif choice == 9:

        lag.findbestAlpha_2(Var_ini = Var_ini,
                            alpha_1=alpha_1,
                            alpha_2=alpha_2,
                            iterations=iterations,
                            K_ref=K_ref,
                            dim_opt=dim_opt,
                            beta=beta)

    elif choice == 10:

        cc.solve_and_plot(alpha=alpha,
                          iterations=iterations,
                          K_ref=K_ref, dim_opt=dim_opt,
                          method_name="projected")

    elif choice == 11:

        cc.solve_and_plot_with_optimal_step(iterations=iterations,
                                            K_ref=K_ref,
                                            dim_opt=dim_opt,
                                            method_name="projected")


