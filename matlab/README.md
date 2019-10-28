This folder contains the matlab code for the project implementing two algorithms: real and complex Schur form RQI for tensor eigenpairs.
The references are:

Newton Correction Methods for Computing Real Eigenpairs of Symmetric Tensors
Ariel Jaffe, Roi Weiss, and Boaz Nadler
SIAM Journal on Matrix Analysis and Applications 2018 39:3, 1071-1094
https://epubs.siam.org/action/showCitFormats?doi=10.1137%2F17M1133312

The number of eigenvalues of a tensor
Dustin Cartwright, Bernd Sturmfels
Linear Algebra and its Applications
Volume 438, Issue 2, 15 January 2013, Pages 942-952
https://www.sciencedirect.com/science/article/pii/S0024379511004629

And our paper in
https://github.com/dnguyend/lagrange_rayleigh/blob/master/docs/LagrangeRayleigh.pdf

Summary: Schur form OCM produces the same iteration as OCM, but solving an equation on $R^n$ instead of $R^{n-1}$, and is faster in general.
Unitary form of Schur form OCM can compute all complex eigenpairs


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnguyend/lagrange_rayleigh/blob/master/EigenTensor.ipynb)

Details:
To compare, we imported the code from Jaffee's Binary Latent Variables
https://github.com/arJaffe/BinaryLatentVariables
the main procedure in that project is orthogonal_newton_correction_method.m
We made a modification to use .' (simple transpose) vs '(complex conjugate transpose) in symmetric_tv_mode_product, to make the
tensor evaluation works for the complex case.

schur_form_rayleigh_raw.m is the implementation of Schur form Rayleigh mirroring OCM.
schur_form_rayleigh.m is the version where we remove one extra tensor calculation.

The main test program for the real case is loop_compare_schur_vs_oncm.m
Since we dont have a script to generate random symmetric tensor in matlab,
we use the one in python (generate_symmetric_tensor.py). The code generates a configuration
of tensors of sizes typical in application and save in test_tensors.mat
loop_compare_schur_vs_oncm.m run thru the tensors and save the results in matlab_save_res.mat

For the complex case, the main function to compute one pair is schur_form_rayleigh_unitary.m
The main function to compute all pairs is find_all_complex_eigen_pairs.m
The main loop to test over all tensors is loop_univary.m
The results are saved in matlab_save_unitary_res.mat.
The python code in check_results_for_complex_pairs() function in generate_symmetric_tensor.py
generates the summary table.

