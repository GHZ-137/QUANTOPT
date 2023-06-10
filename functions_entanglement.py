#####################################################
# Quantum states creation, manipulation and entanglement measurement.
# Create distributions of possibly entangled two-photon polarization states:
# a|VV> + b|VH> + c|HV> + d|HH>
#
# Entanglement (entang) is measured using entanglement entropy:
# Von Neumann entropy of the reduced density matrix, ranging from
# 0 to log(2) = 0.6931
#
# We assume that the distributions for two-photon polarization entanglement is
# Maxwellian or Gaussian
#
# Only general states (pure and partially entangled) are considered here
#####################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm

#####################################################
# Density matrix and entanglement measurements
#
#
#####################################################
def densityMatrix(state, norm_trace = 1):
    ### COMPLEX CONJUGATE
    """
    The density matrix is the outer product of a vector state with itself
    Arguments:
        norm_trace - whether to perform trace normalization (default) or not
    """
    res = np.outer( np.conjugate(state), state)
    if norm_trace:
        res /= res.trace()
    else:
        pass
    return res

def tensorProduct(states):
    """
    Performs Kronecker product of a list of states.
    Arguments:
        states - list of states (density matrices or vector states)
    Tensor product is calculated recursively by multiplying the first two matrices in the list
    and shortening the list.
    """
    #Vector states
    if len(states[0].shape) == 1:
        matrices = [densityMatrix(each) for each in states]
    else:
        matrices = states
    
    if len(matrices)== 2:
        [mat1, mat2] = matrices
        res = np.tensordot(mat1, mat2, axes=0)
        res = res.reshape((mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1]))
        
    else:
        head = [ tensorProduct( [matrices[0], matrices[1]] ) ]
        tail = matrices[2:]                     
        return tensorProduct(head + tail)
    return res

def reducedDensity(mat, final_mat_size = 2, block_size = 0):
    """
    This is the inverse function of tensor product.
    Obtains the reduced trace (reduced density matrix) of the first subsystem in a bipartite system
    by factoring out the second subsystem.
    
    If final_mat_size == 0 and block_size <> 0, calculate final_mat_size
    If final_mat_size <> 0 and block_size ==  0, calculate block_size
    
    The elements of the matrix of size final_mat_size are the trace of consecutive blocks
    of size block_size.

    Arguments:
        mat - density matrix
        final_mat_size - the required final size of the reduced matrix, default 2
        block_size - or the required block size to use, if final_mat_size == 0
        
    """
    if block_size == 0:
        block_size = mat.shape[0] // final_mat_size
        
    elif final_mat_size == 0:
        final_mat_size = mat.shape[0] // block_size

    #print (final_mat_size, block_size)
    
    res = np.zeros((final_mat_size, final_mat_size)).astype(np.cdouble)
    for cont_row in range(0, final_mat_size):
        for cont_col in range(0, final_mat_size):
            res[cont_row, cont_col] = (mat[cont_row* block_size : (cont_row+1)* block_size,\
                                           cont_col* block_size : (cont_col+1)* block_size]).trace()
    
    return res

def SEntropy(state, Schmidt = 1):
    """
    Von Neumann entropy (S) of a density matrix:
        -Tr(rho * log(rho) )
    Arguments:
        mat - state vector or density matrix
        Schmidt - use Schmidt method based on eigenvalues (default) or not
            -SUM( Lambda**2 * log(lambgda**2)
    """
    #Vector state
    if len(state.shape) == 1:
        rho = densityMatrix( np.array(state).T)
    else:
        rho = state
    if Schmidt:
        eigenvals = np.real(np.linalg.eigvals(rho)) + 1e-25
        res = -( np.nan_to_num( eigenvals**2 * np.log(eigenvals**2))  ).sum()
    else:
        res = -( np.matmul( mat, logm(rho))).trace()
    return res

def entang(state, entang_mat_size = 2):
    """
    Entanglement of a two level, bipartite system from its four coefficient state vector.
    1. Calculate density matrix from the state vector
    2. Obtain the reduced density matrix of the first subsystem
    3. Calculate S entropy
    Arguments:
        state - may be a vector state of 4 coefficientes of a two level bipartite system:
            a|VV> + b|VH> + c|HV> + d|HH>
        0r a density matrix.
        entang_mat_size - reduced density matrix size.
        
    If a vector is given, the 2x2 reduced density matrix is calculated.
    If a density matrix is given, it is iteratively reduced to 2x2   
    """    
    #Vector state
    if len(state.shape) == 1:
        rho = densityMatrix( np.array(state).T)
    else:
        rho = state
        
    rho_reduced = reducedDensity(rho, final_mat_size = entang_mat_size)        
    return SEntropy(rho_reduced)

def mutualInformation(states, entang_mat_size = 2):
    """
    Mutual information from several, two level, bipartite system.
    Arguments:
        states - list of state vectors or density matrices.
        entang_mat_size - reduced density matrix size.
    """
    rho_composite = tensorProduct( states )
    indiv_S = [SEntropy(each) for each in states]
    res = np.array(indiv_S).sum() -SEntropy(rho_composite)
    return res 

#####################################################
# Bell states
#
#####################################################
# Define Bell states and density matrices

PHI_P = np.array([1./np.sqrt(2),0,0, 1./np.sqrt(2)])
PHI_N  = np.array([1./np.sqrt(2),0,0, -1./np.sqrt(2)])
PSI_P  = np.array([0, 1./np.sqrt(2), 1./np.sqrt(2), 0])
PSI_N  = np.array([0, 1./np.sqrt(2), -1./np.sqrt(2), 0])

PSI_P_mat = densityMatrix(PSI_P)
PSI_N_mat = densityMatrix(PSI_N)
PHI_P_mat = densityMatrix(PHI_P)
PHI_N_mat = densityMatrix(PHI_N)

BELL_MATRICES = [PSI_N_mat, PSI_P_mat, PHI_P_mat,PHI_N_mat ]
SINGLET = PSI_N_mat; PSI_N = SINGLET
NON_SINGLET = PSI_P_mat + PHI_P_mat + PHI_N_mat
PHI_POS = PHI_P_mat
NON_PHI_POS = PSI_P_mat + PSI_N_mat + PHI_N_mat
PSI_POS = PSI_P_mat
NON_PSI_POS = PSI_N_mat + PHI_P_mat + PHI_N_mat

def Bell_diag_state(probs):
    """
    Returns a Bell state density matrix from a list of mixing probability values
    Arguments:
        probs - 4 element array containg he mixing probabilities
        for each Bell state
    """
    state = probs[0] * BELL_MATRICES[0]
    for cont in range(1,4): 
        state += probs[cont] * BELL_MATRICES[cont]
    return state

def Werner_Psi_neg(F):
    """
    Returns a Werner Psi neg. state density matrix from a fidelity value (F).
    Arguments:
        F - fidelity value
    """
    other_p = (1.-F)/ 3.
    state = F * SINGLET + ((1.-F)/ 3.) * NON_SINGLET
    return state

def Werner_Psi_pos(F):
    """
    Returns a Werner Psi pos. state density matrix from a fidelity value (F).
    Arguments:
        F - fidelity value
    """
    other_p = (1.-F)/ 3.
    state = F * PSI_POS + ((1.-F)/ 3.) * NON_PSI_POS
    return state

def Werner_Phi_pos(F):
    """
    Returns a Werner Phi pos.state density matrix from a fidelity value (F).
    Arguments:
        F - fidelity value
    """
    other_p = (1.-F)/ 3.
    # Isotropic state
    # IDENT = np.identity(4) * 0.25
    #state = F * PHI_POS + ((1.-F) ) * IDENT
    state = F * PHI_POS + ((1.-F) ) * NON_PHI_POS
    return state


def fidelity(state, target = PSI_N):
    """
    Measures fidelity (F) of a state  vector or matrix to a target state (default singlet).
    Arguments:
        state - state vector or density matrix or state vector
        target - state target vector
    """
    if len(state.shape) == 1:
        state = densityMatrix(state)
      
    right_mul = np.matmul(state, target)
    F = np.matmul(target, right_mul)
    return F

def Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 ):
    """
    Bell state entanglement between several particles
    Arguments:
        diag - probs. as diag of density matrix
        pos - indices of positive and negative probs.
        for V and H cases.
    """
    
    p1 = np.array( [diag[each] for each in pos_1] )
    p2 = np.array( [diag[each] for each in pos_2] )
    pn1 = np.array( [diag[each] for each in pos_neg_1] )
    pn2 = np.array( [diag[each] for each in pos_neg_2] )
    sum1 = np.sum(p1) + np.sum(pn1)
    sum2 = np.sum(p2) + np.sum(pn2)
    p1 /= sum1
    pn1 /= sum1
    p2 /= sum2
    pn2 /= sum2
    #n_part = len(pos_1) + len(pos_2)
    #res =[ np.sum(p1) -np.sum(pn1),  np.sum(p2) -np.sum(pn2)]
    E1 = 4 * (np.prod(p1) - np.prod(pn1))
    E2 = 4 * (np.prod(p2) - np.prod(pn2))


    res = np.array([E1, E2])
    ####res = np.clip(res, 0., float('Inf'))

    return res
    
