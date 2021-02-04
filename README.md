This directory provides an implementation of the algorithms for dynamic assortment optimization under the Multinomial logit choice model.

## Dependencies

The package dependencies of the code are latest versions of the following packages:
* `numpy`
* `pandas`
* `gurobipy`
* `scipy`
* `numba`



## Functional description

The object constructed with `Numerical()` instanciates the random generator of the simulation.
* The method `refresh_data()` samples new inputs,
* The method `sample_revenue()` uses a sampling-based estimator for the expected revenue,
* Several algorithmic methods are implemented: our constant-factor approximation `algorithm()`, a discrete greedy heuristic `greedy()`, a greedy local search method `local_search()`, a gradient-descent with regards to the Lovasz extension of the expected revenue function `lovasz_greedy()`, a dynamic programming-based relaxation `relaxation_topaloglu()`, an MIP-based deterministic relaxation `relaxation_deterministic()`.
* Other functions are auxiliary and need not be referenced in running the code.

The file `Numericals.py()` can be run as a script to replicate the computational results in the paper *Greedy-Like Algorithms for Dynamic Assortment Planning Under Multinomial Logit Preferences*, A. Aouad, R. Levi and D. Segev (2018) that appeared in Operations Research.

A work-in-progress implementation of a polynomial-time approximation scheme is provided in the file `Ptas.py`.
