Author: Du Nguyen

This project contains supporting codes for the paper [Lagrange Multipliers and Rayleigh Quotient Iteration in Constrained Type Equations](https://github.com/dnguyend/lagrange_rayleigh/blob/master/docs/LagrangeRayleigh.pdf).

To run clone the project as a subfolder of some folder folder (say <parent>). You will see a folder <parent>/lagrange_rayleigh. Add <parent> to your PYTHONPATH and then run the codes in <parent>/lagrange_rayleigh/test/
Python 3 but should be mostly compatible with python 2.7 Require numpy and scipy

We show an example where the method here provide an improvement to the eigen-tensor problem by 30% for the real case.
We also show it leads to an algorithm to compute all complex eigenpairs (counted by Cartwright-Sturmfels formula):
https://www.sciencedirect.com/science/article/pii/S0024379511004629

The notebook demonstrating this algorithm can be found here.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/EigenTensor.ipynb)
