"""
Optic basis functions for multipartite states
 a|VVV> + b|VVH>  + c|VHV>  + d|VHH> + ...

represented as complex coefficient vectors or matrices of size:
 d^n (d- levels, n- parts)
"""

import numpy as np
from copy import deepcopy
from numpy.linalg import svd,norm

from functions_entanglement import *
from functions_basis import *

from config_quantum import *
from config_experiment import *


#######################################################
# State functions
#######################################################
# Matrix of magnitude of a complex matrix
def mag_mat(mat):
    r = np.real(mat)
    i = np.imag(mat)
    res = np.sqrt(r**2 + i**2)
    return res

# Normalize state by trace = 1
def norm_state(state):
    state.flags.writeable = True
    if len(state.shape)>1:
        t = mag_mat(state).trace()
        state /= t
    else:
        
        den = np.sqrt( np.sum( mag_mat(state)**2 ))
        if den !=0:
            state *= 1./ den
       
    return state

# Create empty complex matrix
def vacuum_state(n=2):
    global d
    res = np.zeros(d**n).astype(np.cdouble) #csingle) #
    return res

# States
def part_ent_state(F):
    """ Quantum optics part. ent. state: Phi pos(F) + Psi neg(1-F) """
    res = F * (PSI_POS) + ((1.-F) ) * (PHI_N_mat)
    ###res = F * (PHI_POS) + ((1.-F) ) * (PSI_N)
    return res
        
def superp_state(Theta = 45.):
    Theta_deg = np.pi * Theta / 180.
    state = vacuum_state(1)
    state[0] =  np.cos(Theta_deg)
    state[1] =  np.sin(Theta_deg)
    state = np.outer(state, state)
    return state

# Discard imaginary residues from diag
def probs(state):
    return np.real( np.diag(state))

# Number of parts from a matrix of size(len) assuming global d levels
def calc_n_parts(mat):
    global d
    l = len(mat)
    res = np.log(l) / np.log(d)
    res = int(np.round(res))
    return res

#######################################################
# Optical toolkit
#
#######################################################
# Phase shifter for a 2-state particle (2x2 matrix)
def phase_op(Phi_deg = 45.):
    Phi = np.pi * Phi_deg / 180.
    res = np.outer(vacuum_state(1), vacuum_state(1))
    e = np.exp((0.+1.j) * (Phi) )
    #res[:] =  [[1., e1],[e1, 1.]]
    res[:] =  [[1, 0.],[0., e]]
    return res

# Meta-operator to tensorize a 2x2 operator (matrix) for n_parts
def tensorize_op(op, n_parts):
    # Duplicate the original operator!
    op_ori = deepcopy(op)
    for cont in range(n_parts - 1):
        op = np.kron(op, op_ori)
    return op

# Application of a 2x2 operator to a n_parts state matrix
def apply(state, op):
    # Tensorize operator
    n_parts = calc_n_parts(state)
    op = tensorize_op(op, n_parts)
    
    # Unitary evolution over a density matrix: U * rho * U_dagger
    res = np.matmul(op, state)
    res = np.matmul(res, np.conjugate(op.T))
    norm_state(res)
    return res

# Phase shifter operator of a state with parameter:
# Phi: phase shift
def PHS(state, Phi_deg = 0.):
    # Apply operator
    res = apply(state, phase_op(Phi_deg) )
    norm_state(res)
    return res
    

# Rotation matrix used in beam splitter
def trig_mat(Theta_deg = 45., Phi_deg = 0.):
    Theta = np.pi * Theta_deg / 180.
    Phi = np.pi * Phi_deg / 180.
    c = np.cos(Theta)
    s = np.sin(Theta)
    e = np.exp((0.+1.j) * (Phi) )
    e_neg = np.exp((0.+1.j) * (-Phi) )
    res = np.outer(vacuum_state(1), vacuum_state(1))
    res[:] = [[c, -e * s], [e_neg * s,c]]
    return res

# Beam splitter operator  of two states with parameters:
# Theta: reflec. and transm. rate [45º -> 50:50]: (-180º, 180º)
# Phi: phase shift
def BS(state_A, state_B, Theta_deg = 45., Phi_deg = 0.):
    # Modes product
    res = np.kron(state_A, state_B) + np.kron(state_B, state_A)
    res1 = np.kron(state_A, state_B)
    res2 = np.kron(state_B, state_A)

    # Trigonometric matrix
    t_mat = trig_mat(Theta_deg, Phi_deg)
    t_mat1 = trig_mat(Theta_deg, Phi_deg)
    t_mat2 = trig_mat(90. - Theta_deg, 0) #-Phi_deg) <===
 
    # Apply operator
    res = apply(res, t_mat)
    res1 = apply(res1, t_mat1)
    res2 = apply(res2, t_mat2)
    res = res1 + res2 # .T  #need to transpose?
    
 
    norm_state(res)
    return res 

############################
## TESTS
############################
"""
a = superp_state(35.)
b = part_ent_state(.9)
apply(a, phase_op())
apply(b, phase_op())
np.diag(BS(a, b)).sum()
np.diag(BS(a, b, 45, -37)).sum()
probs(BS(a, b, 45, -37))
"""
