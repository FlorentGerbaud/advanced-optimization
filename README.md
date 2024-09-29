# Advanced-optimization

# README for Heat Source Optimization Project

## Overview

This project implements a finite element analysis framework to optimize heat sources in a one-dimensional domain. The objective is to adjust the heat sources so that the temperature distribution within the domain closely matches a predefined target temperature profile.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Functions](#functions)
- [Optimization Report](#optimization-report)
  - [Introduction](#introduction)
  - [Heat Conduction and Simulation](#heat-conduction-and-simulation)
  - [Gradient Descent Optimization](#gradient-descent-optimization)
  - [Results](#results)
    - [Comparison of Optimization Methods](#comparison-of-optimization-methods)
    - [Visualizations](#visualizations)
  - [Conclusion](#conclusion)
- [License](#license)

## Requirements

- Python 3.x
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/FlorentGerbaud/advanced-optimization.git
```
```bash
cd advanced-optimization
```

## Usage

To run the optimization, execute the following command:

```bash 
python heat_bar_starting.py
```

This will execute the simulation, compute the initial temperature distribution, and generate plots comparing the initial and target temperature profiles.

## Code Structure

**Main Script:** Contains the core logic for defining parameters, computing conduction coefficients, and simulating the temperature distribution.

### Functions:
- **`compute_conduction(noise_val):`** Computes the conduction coefficients with optional noise.
- **`performBernstein(x, Var):`** Evaluates the Bernstein polynomial for given design variables.
- **`compute_source(Var_opt):`** Computes the heat source based on the optimized variables.
- **`compute_matrix(K):`** Constructs the finite element matrix using the conduction coefficients.
- **`compute_rhs(S):`** Constructs the right-hand side of the finite element equations.
- **`simulator(noise, Var):`** Runs the finite element simulation.
- **`costFunction():`** Calculates the cost function to assess the deviation from the target temperature.
- **`compute_adjoint(T, T_star, K):`** Solves the adjoint equation to compute the Lagrange multiplier lambda.
- **`compute_gradient(lambda_adj, Var_opt):`** Computes the gradient of the cost function with respect to the design variables.
- **`gradient_descent(Var_opt, alpha, iterations):`** Performs gradient descent to minimize the cost function.
- **`line_search(u, grad_u, cost_func):`** Performs line search to find the optimal step size alpha.
- **`gradient_descent_with_line_search(Var_opt, iterations):`** Performs gradient descent with optimal step size determined by line search.

### Functions

### Discretization Parameters
- **`l:`** Domain length (set to 1).
- **`n_el:`** Number of finite elements (set to 200).
- **`h:`** Element size.
- **`n_no:`** Number of nodes.

### Distributions
- **`T_star:`** Target temperature distribution defined as
  $$
  T_{star} = 1 + \left(\frac{X}{l}\right)^2
  $$

### Initial Configuration
- **`Var_ini:`** Initial design vector for heat sources, initialized to zero.

### Equations

#### Heat Equation
The heat equation is given by:

$$
\frac{d}{dx}\left(k(x) \frac{dT}{dx}\right) = S(x)
$$

where $ k(x) $ is the conduction coefficient, $ T(x) $ is the temperature, and \( S(x) \) is the heat source.

#### Cost Function
The cost function $ J(T) $ is defined as:
$$
J(T) = \frac{1}{2} \int_{0}^{l} (T(x) - T^*(x))^2 \, dx
$$

#### Lagrangian
The Lagrangian $ L $ is defined as the sum of the cost function and the heat equation multiplied by the Lagrange multiplier \( \lambda(x) \):
$$
L(T, \lambda) = J(T) + \int_{0}^{l} \lambda(x) \left( \frac{d}{dx}\left(k(x) \frac{dT}{dx}\right) - S(x) \right) \, dx
$$

#### Adjoint Equation
The adjoint equation is derived by setting the derivative of the Lagrangian with respect to $ T $ to zero:
$$
\frac{\delta L}{\delta T} = 0
$$
This results in:
$$
\begin{cases}
\frac{d}{dx}\left(k(x) \frac{d\lambda}{dx}\right) = - (T(x) - T^*(x)) \\
\lambda(x) \cdot k(x) \cdot \nabla T = 0 \text{ at boundaries} \\
\frac{d}{dx}\left(\lambda(x) \cdot k(x) \right) \cdot T = 0 \text{ at boundaries}
\end{cases}
$$


#### Gradient Calculation
The gradient of the cost function with respect to the design variables $ Var_{opt} $ is given by:

$$
\frac{\delta J}{\delta Var_{opt}[i]} = -\int_{0}^{l} \lambda(x) B_i(x) \, dx
$$

where $ B_i(x) $ is the Bernstein polynomial of degree $ dim\_opt - 1 $:

$$
B_i(x) = \binom{dim\_opt - 1}{i} x^i (1-x)^{dim\_opt - 1 - i}
$$

#### Discretization
The integral is discretized using the method of rectangles:
$$
\nabla_i = -\int_{0}^{l} \lambda(x) B_i(Xg(x)) \, dx \approx -\sum_{j=0}^l \lambda(j) \cdot B_i(Xg[j]) \cdot h
$$
where $ h $ is the element size.

## Optimization Report

### Introduction

The objective of this program is to solve a 1D heat conduction problem using optimization methods. The problem involves adjusting a heat source function, modeled by Bernstein polynomials, so that the simulated temperature matches a target temperature $ T^{\star} $.

The optimization is performed using gradient descent, with two strategies: one with a fixed step size and the other with optimal step size search (line search). The adjoint equation is used to compute the gradient of the cost function, which measures the difference between the simulated and target temperatures.

### Heat Conduction and Simulation

#### Physical Problem

The simulation of the heat conduction problem is based on a finite element discretization model. Thermal conductivity is modeled by a function $ K(x) $ that can include noise to simulate uncertainties. The heat source term is expressed by Bernstein polynomials parameterized by an optimization variable vector `Var_opt`.

The main conduction equation is solved in the form $ M \cdot T = \text{RHS} $, where $ M $ is the system matrix, $ T $ is the temperature, and `RHS` is the right-hand side corresponding to the heat source.

#### Main Functions

- **`compute_conduction(noise_val)`**: generates the thermal conductivity function with or without noise.
- **`performBernstein(t, Var_opt)`**: evaluates a Bernstein polynomial based on the optimization variables.
- **`compute_matrix(K)`**: builds the system matrix based on conductivity \( K(x) \).
- **`compute_rhs(S)`**: generates the system's right-hand side based on the heat source \( S(x) \).
- **`simulator(noise, Var)`**: solves the system based on the optimization variables to compute the temperature.

### Gradient Descent Optimization

The optimization adjusts the coefficients `Var_opt` of the Bernstein polynomials to minimize the difference between the simulated temperature and the target temperature. This difference is measured by a cost function.

#### Cost Function

The cost function is defined as:

$$
J(T) = \frac{1}{2} \sum (T - T^{\star})^2 h
$$

where $ T^{\star} $ is the target temperature, $ T $ is the simulated temperature, and $ h $ is the element size.

#### Adjoint Equation

The adjoint equation allows the calculation of the gradient of the cost function with respect to the optimization variables `Var_opt`. The method involves solving a problem similar to the direct problem, but with modified boundary conditions.

The gradient of the cost function is then computed using the solution of the adjoint equation, allowing the optimization variables to be updated during gradient descent.

#### Gradient Descent

Two gradient descent methods are implemented:

- **Gradient descent with fixed step**: A constant learning rate $ \alpha $ is used to adjust the optimization variables at each iteration.
- **Gradient descent with line search**: The optimal step size is determined at each iteration through a line search, improving convergence.

The implemented functions for gradient descent are:

- **`compute_adjoint(T, T_star, K)`**: solves the adjoint equation to calculate the Lagrange multiplier $ \lambda $.
- **`compute_gradient(lambda_adj, Var_opt)`**: computes the gradient of the cost function based on $ \lambda $ and the optimization variables.
- **`gradient_descent(Var_opt, alpha, iterations)`**: runs gradient descent with a fixed step size.
- **`line_search(u, grad_u, cost_func)`**: performs the line search to find the optimal step size.
- **`gradient_descent_with_line_search(Var_opt, iterations)`**: runs gradient descent with line search.

### Results

#### Comparison of Optimization Methods

The two methods (fixed step and line search) are compared over 10,000 iterations to adjust the initial optimization variables `Var_opt`.

- **Gradient descent with fixed step**:
    - Optimized variables: `Var_optWithFixStep`
    - Computation time: `fixed_step_time` seconds
    - Final error: `error_fixed_step`

- **Gradient descent with line search**:
    - Optimized variables: `Var_optWithOptimalStep`
    - Computation time: `optimal_step_time` seconds
    - Final error: `error_optimal_step`

#### Visualizations

1. **Conductivity coefficient**: A plot of $ K(x) $ is shown to visualize the variation of conductivity across the domain.
   
2. **Error comparison**: A log-log plot is generated to compare the convergence of errors for both optimization methods.

3. **Temperature profiles**: A plot displays the initial, target, and optimized temperature profiles after 10,000 iterations.

### Conclusion

The method with line search shows better convergence than the fixed step method. Although the line search takes more time per iteration, it reaches a solution faster in terms of error relative to the target temperature. The adjoint formulation allowed efficient optimization by directly computing the gradient of the cost function.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
