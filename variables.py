#_______________________________________ import libraries _______________________________________#

import numpy as np

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