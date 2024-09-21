# advanced-optimization

# README for Heat Source Optimization Project

## Overview

This project implements a finite element analysis framework to optimize heat sources in a one-dimensional domain. The objective is to adjust the heat sources so that the temperature distribution within the domain closely matches a predefined target temperature profile.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Functions](#functions)
- [Results](#results)
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

- **Main Script:** Contains the core logic for defining parameters, computing conduction coefficients, and simulating the temperature distribution.

### Functions:
- **`compute_conduction(noise_val):`** Computes the conduction coefficients with optional noise.
- **`performBernstein(x, Var):`** Evaluates the Bernstein polynomial for given design variables.
- **`compute_source(Var_opt):`** Computes the heat source based on the optimized variables.
- **`compute_matrix(K):`** Constructs the finite element matrix using the conduction coefficients.
- **`compute_rhs(S):`** Constructs the right-hand side of the finite element equations.
- **`simulator(noise, Var):`** Runs the finite element simulation.
- **`costFunction():`** Calculates the cost function to assess the deviation from the target temperature.

## Functions

### Discretization Parameters
- **`l:`** Domain length (set to 1).
- **`n_el:`** Number of finite elements (set to 200).
- **`h:`** Element size.
- **`n_no:`** Number of nodes.

### Distributions
- **`T_star:`** Target temperature distribution defined as 
  \[
  T_{star} = 1 + \left(\frac{X}{l}\right)^2
  \]

### Initial Configuration
- **`Var_ini:`** Initial design vector for heat sources, initialized to zero.

## Results

Upon running the simulation, the program generates plots comparing the initial temperature distribution (`T_ini`) and the target temperature distribution (`T_star`). The results will help visualize how well the current configuration approaches the desired thermal state.

## License

This project is licensed under the MIT License - see the LICENSE file for details.