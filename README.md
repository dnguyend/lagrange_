Author: Du Nguyen

This project contains supporting codes for an *older version* of the paper [Lagrange Multipliers and Rayleigh Quotient Iteration in Constrained Type Equations](https://github.com/dnguyend/lagrange_rayleigh/blob/master/docs/LagrangeRayleigh.pdf).

# A new version is coming out - stay tuned.

To run clone the project as a subfolder of some folder folder (say <parent>). You will see a folder <parent>/lagrange_rayleigh. Add <parent> to your PYTHONPATH and then run the codes in <parent>/lagrange_rayleigh/test/
Python 3 but should be mostly compatible with python 2.7 Require numpy and scipy

We show an example where the method here provide an improvement to the eigen-tensor problem by 30% for the real case.
We also show it leads to an algorithm to compute all complex eigenpairs (counted by Cartwright-Sturmfels formula):
https://www.sciencedirect.com/science/article/pii/S0024379511004629

The matlab code for the eigentensor algorithm can be found in the matlab folder.
The other notebooks demonstrating this algorithm can be found here - to run online on Google colab you can click on the link:
Eigentensor
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/EigenTensor.ipynb)

Nonlinear Eigenvalue:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/NonLinearEigen.ipynb)

Eigenvector:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/Eigen.ipynb)

Two Left Inverses: show that we can use one left inverse for lambda and another for projection.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
(https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/TwoLeftInverses.ipynb)
