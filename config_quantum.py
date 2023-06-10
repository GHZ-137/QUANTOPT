import numpy as np
from functions_basis import *

## Basis config
# Constants and vector representation
V = 'V'
H = 'H'
_V = np.array([1., 0.])
_H = np.array([0., 1,])

# Basis definition: num. of levels
basis= [V,H]
d = len(basis)

# States and gates quality: 95, 97, 99
quality = 0.95

## Parameters for state generation function
# 2-states: Werner fidelity
F_func = 'Gaussian' #Uniform
F_mean = quality
F_sigma = 0.
F_lim = [.875, .999]

# 1-states: Theta
single_func = 'Gaussian'
single_mean = 180 * np.arcsin(quality /np.sqrt(2)) / np.pi
single_sigma = 0.
single_lim = [35, 55]

# Added noise definition
noise_func = 'Gaussian'
noise_mean = 0.
noise_sigma = 0.
noise_lim = [-1, 1]

##  System config
# Tools functions
Theta_func = 'Uniform' #'Gaussian'
Theta_mean = 45.
Theta_sigma = 0.
Theta_lim = [0.1, 90] #[-180., 180.]

Phi_func = 'Uniform' #'Gaussian'
Phi_mean = 45.
Phi_sigma = 0. #1.
Phi_lim = [0, 90] # [-90, 90]

p_extra_PHS = 0.1 # .1  .5 #PHS prob << 0.1
p_BS_PHS_change = 0.9 #0.5 ##.5 # PHS different to zero in BS << 0.9
