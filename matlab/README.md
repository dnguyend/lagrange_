This folder contains the matlab code for the project implementing two algorithms: real and complex Schur form RQI for tensor eigenpairs.
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

