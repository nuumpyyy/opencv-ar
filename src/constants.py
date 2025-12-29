import numpy as np

# Original camera matrix obtained in calibration script
IN_MTX_ORIGINAL = np.array([[3093.96586, 0, 1531.99079],
                             [0, 3091.13832, 2013.36704],
                             [0, 0, 1]])

# Refined camera matrix
IN_MTX_OPTIMAL = np.array([[3110.46091, 0, 1525.90159],
                           [0, 3119.78584, 2009.91548],
                           [0, 0, 1]])

# Distortion coefficients calculated in script
DIST_COEFFS = np.array([[0.249932046, -1.29011763, -0.00101717369, -0.00243230478, 1.78343001]])

#print(IN_MTX_ORIGINAL, IN_MTX_OPTIMAL, DIST_COEFFS)
#print(np.shape(IN_MTX_ORIGINAL), np.shape(IN_MTX_OPTIMAL), np.shape(DIST_COEFFS))