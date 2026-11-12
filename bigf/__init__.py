"""Numerical companion code for block-isotropic Gaussian fields.

Modules
-------
profiles     Covariance profiles h and their derivatives.
bounds       Variance envelope, error thresholds and probability bounds.
inputs       Gaussian input model used for the joint (field and input) bounds.
covariance   Covariances of the field, its quadratic Taylor approximation and the error.
sampling     Point sets, the variance-maximizing pair and sampling of approximation errors.
calibration  Coverage statistics for a threshold against sampled errors.
fixed_input  The fixed-input Monte Carlo experiment (Table 1 and Figure 6).
plotting     Figure styles and saving.
paths        Output locations inside the repository.
"""
