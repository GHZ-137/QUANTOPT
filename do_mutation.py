"""
GP (sub-tree mutation) for quantum optics circuit of n parts.

Sources: Part.(Theta), superp.(Theta)
Tools: BS(Theta)(2), HWP(Theta)(1), Brewster(Theta)(1) => Not implemented

"""

from functions_basis import *
from functions_states import *
from functions_quantum_trees import *
from functions_plot import *

from config_quantum import *
from config_GP import *
"""
# Return a HWP random angle according to config
def Theta_HWP():
    global HWP_func, HWP_mean, HWP_sigma, HWP_lim
    if HWP_func == 'Gaussian':
        res = np.random.normal(HWP_mean, HWP_sigma)
        np.clip(res, HWP_lim[0], HWP_lim[1])
    if HWP_func == 'Uniform':
        res = HWP_lim[0] + np.random.rand() * (HWP_lim[1] - HWP_lim[0])
    if HWP_func == 'Classic':
        # H, 45º or V
        res = 45. * np.random.randint(3)        
        #res = (np.random.rand() * 80 ) + 10
    return res

# Raturn a HWP random angle according to config
def Theta_BS():
    global BS_func, BS_mean, BS_sigma, BS_lim
    if BS_func == 'Gaussian':
        res = np.random.normal(BS_mean, BS_sigma)
        np.clip(res, BS_lim[0], BS_lim[1])
    if BS_func == 'Uniform':
        res = BS_lim[0] + np.random.rand() * (BS_lim[1] - BS_lim[0])
    return res
  
# Random integer decompositions of n parts into single and double source states
def rand_decomp(n):
    if n == 1:
        res = [1]
    elif n == 2:
        a = [ [ 1,1 ], 2 ]
        # change probability!
        res = [ a[np.random.choice([0,1])] ]
    else:
        cont1 = n-1
        cont2 = 1
        res = []
        while (cont1 >= cont2):
            res.append([rand_decomp(cont1), rand_decomp(cont2)])
            cont1 -=1
            cont2 +=1
    return res[ np.random.choice( range(len(res)) )]

# Make circuit from a random decomposition with global p of extra HWP
def circuit(decomp):
    ## 1º. LISTA EXTRAS SI NO ES NODO SUPERIOR *
    ## 2º. VALOR DETECTOR SI DECOMP= 1, SI PRIMERO PUEDE TENER dos
    ## 3º. BS, PBS
    
    global p_extra_HWP, p_extra_Pauli
    extras = 'None'
    if np.random.rand() < p_extra_HWP:
        extras = ' HWP_' + str( np.round(Theta_HWP(), 2 ))
    #if np.random.rand() < p_extra_Pauli:
    #    c = ['X','Y','Z'][np.random.randint(3)]
    #    extras = ' Pauli_' + c

    # Terminal nodes
    if decomp == 1:
        res =  [extras + ' Sup._' + str( np.round(Theta(), 2 ))]
        c = 'None'
        res[0] = c +' ' + res[0]
        return res
    
    if decomp == 2 :
        res = [extras + ' Part._' + str(np.round(F(), 2 ))]
        l = ['None', 'Detect', 'Herald']
        c = l[ np.random.randint(3)]
        ##c = 'Detect'
        res[0] = c +' ' + res[0]
        return res
        
    if len(decomp) == 2:
        extras = 'None'
        if np.random.rand() < p_extra_HWP:
            extras = ' HWP_' + str( np.round(Theta_HWP(), 2 ))
        #if np.random.rand() < p_extra_Pauli:
        #    c = ['X','Y','Z'][np.random.randint(3)]
        #    extras = ' Pauli_' + c
        
        l = ['BS', 'PBS']
        c = 'BS' #l[ np.random.randint(2)]
        res =  [extras + ' BS', [ circuit(decomp[0]), circuit(decomp[1])]  ]


        l = ['None', 'Detect', 'Herald']
        c = l[ np.random.randint(2)]
        ##c = 'Detect'
        res[0] = c +' ' + res[0]
        
        return res

# Fitness function
def f_function(target, actual):
    global alpha_zero
    idx = list(np.where(target !=0)[0])
    a = 0
    b = 0
    for each in range(len(target)):
        if each in idx:
            a += actual[each]
        else:
            b += actual[each]
    #res = a / b
    #res = np.corrcoef(target, actual)[0][1]
    #zeros = len( np.where( (target - actual) == 0 )[0] )
    #res += alpha_zero * zeros
    #res = np.sum(-np.log((target - actual)**2))
    #res =  np.sum(target * actual)
    ##res = np.corrcoef(target, actual)[0][1]
    #res = np.sum( target * np.log(actual))

    sse = np.sum((target - actual)**2)
    ent = Ent_2x2(actual, pos_1, pos_2, pos_neg_1, pos_neg_2)
    ent = ent[0] + ent[1]
    
    return sse

def print_probs(circ):
    res = evaluate(circ)
    diag = np.real(np.diag(res))
    print('P:' + '%.2f '* len(diag) % tuple(diag) )
    return
"""
def prob_target(circ):
    res = evaluate(circ)
    diag = np.real(np.diag(res))
    pos = 0.; neg = 0.
    for cont in range(len(diag)):
        if target.flatten()[cont] != 0:
            pos += diag[cont]
        else:
            neg += diag[cont]
    res = pos
    return res

# Evaluate the same circuit with different noisy source states
def evaluate_multi(tree):
    global target, repetitions, optimization
    #print('Solving circuit...')
    best_val = float('inf')
    best = None
    fits = []
    diags = []
    for cont in range(repetitions):
    # Set same seed for each repetition number
        ##np.random.seed(rep_seeds[cont])
        res = evaluate(tree)
        diag = np.real(np.diag(res))
        fit = f_function(target, diag)
        #print(fit)
        
        if optimization == 'Minimize':
            if fit < best_val:
                best_val = fit
                best = diag
        else:
            if fit > best_val:
                best_val = fit
                best = diag
            
        fits.append(fit)
        diags.append(diag)
    # Reset run_seed
    ##np.random.seed(seed)
    return (fits, diags)

##################################################################
# Main flow
##################################################################
run = 0
runs = 5
seed = 24
np.random.seed(seed) ## Change seed of seeds
seeds = ((np.random.rand(runs)*1e6)+100).astype(np.int)
#rep_seeds = ((np.random.rand(repetitions)*1e6)+100).astype(np.int)
np.random.seed(seeds[run])

# Variables
from config_experiment import *
"""
dec_n = 2 #number of decimals
n_parts = 5 #8 #7 10 12 14
repetitions = 1 #1 #10
max_ite = 100 #0
alpha_zero = 0. #1e-4
"""

# Save 
output_folder = './test/' #'./output_n_6/'
labels = ['Iter.', 'Best']
labels_f =  "  {: <10} {: <10}"
condition = 'mutation_' + str(run+1)
name = output_folder + '/' + condition + '.txt'
f = open(name, 'w')
cad = ''
for each in labels[:]:
    cad += each + '\t'
f.write( cad + '\n')

"""
# Target state
states = enum_states(n_parts)
target = np.zeros(2**n_parts)

target[0] = 1
target[3] = 1

#Take out for n = 5
#target[-1] = 1
#target[-4] = 1

#target[7] = 1
#target[-8] = 1

#target[7] = 1
#target[-8] = 1
target /= np.sum(target)
"""

# Initial circuit
decomp = rand_decomp(n_parts)
print('Decomp. of '+ str(n_parts) + ':\n', decomp)
circ = circuit(decomp)
circ_idx = tree_indices(circ)

if optimization == 'Minimize':
    res = float('inf')
else:
    res = -float('inf')   
diags = None
for ite in range(max_ite):
    sub_idx = np.random.choice(len(circ_idx))
    sub_n = n_sources(tree_by_idx( circ, circ_idx[sub_idx] ) )
    sub_decomp = rand_decomp(sub_n)
    sub_circ = circuit(sub_decomp)
    new_circ = deepcopy(circ)
    substitute(new_circ, circ_idx[sub_idx], sub_circ)

    fits, new_diags = evaluate_multi(new_circ)
    new_res = np.mean(fits)
    print('%d. Fit: %.4f' % (ite,res) )
    f.write('%d\t%.4f\n' % (ite,res) )

    #  Max-Min
    if optimization == 'Minimize':
        if new_res < res:
            res = new_res
            diags = new_diags
            circ = new_circ
            circ_idx = tree_indices(circ)
    else:
        if new_res > res:
            res = new_res
            diags = new_diags
            circ = new_circ
            circ_idx = tree_indices(circ)


diag = np.mean(np.array(diags), axis = 0)
size = int(np.sqrt(len(diag)))
ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
print('E:')
for each in ent: print('%.4f\t' % each)

# Odd n_parts
if (n_parts % 2 != 0):
    #diag2 = -0.01 * np.ones((size+1)**2)
    diag2 = np.zeros((size+1)**2)
    diag2[: len(diag)] = diag
    diag = diag2
    target2 = np.zeros((size+1)**2)
    target2[: len(target)] = target
    target = target2
    size = np.int(np.sqrt(len(diag)))

diag = diag.reshape((size,size))
title = 'Mutation. Fit:' + ('%.4f' % res) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])
plt_bar3d(output_folder + '/' + condition , diag.T, title, z_lim = 0.3)

#Save target
"""
target = target.reshape((size,size))
plt_bar3d(output_folder + '/target', target.T, 'Target.', z_lim = 0.5)
"""
f.close()
##############################################
# Genetic programming: sub-tree mutation
##############################################
alpha_zero = 0. #1e-4
for ite in range(max_ite):
    sub_idx = np.random.choice(len(circ_idx))
    sub_n = tree_parts(tree_by_idx( circ, circ_idx[sub_idx] ) )
    sub_decomp = rand_decomp(sub_n)
    sub_circ = circuit(sub_decomp)
    new_circ = deepcopy(circ)
    substitute(new_circ, circ_idx[sub_idx], sub_circ)

    fits, new_diags = evaluate_multi(new_circ)
    new_res = np.mean(fits)
    print('%d. Fit: %.4f' % (ite,res) )
    f.write('%d\t%.4f\n' % (ite,res) )

    #  Max-Min
    if optimization == 'Minimize':
        if new_res < res:
            res = new_res
            diags = new_diags
            circ = new_circ
            circ_idx = tree_indices(circ)
    else:
        if new_res > res:
            res = new_res
            diags = new_diags
            circ = new_circ
            circ_idx = tree_indices(circ)

diag = np.mean(np.array(diags), axis = 0)
size = int(np.sqrt(len(diag)))
ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
print('E:')
for each in ent: print('%.4f\t' % each)

# Odd n_parts
if (n_parts % 2 != 0):
    #diag2 = -0.01 * np.ones((size+1)**2)
    diag2 = np.zeros((size+1)**2)
    diag2[: len(diag)] = diag
    diag = diag2
    target2 = np.zeros((size+1)**2)
    target2[: len(target)] = target
    target = target2
    size = np.int(np.sqrt(len(diag)))
    
diag = diag.reshape((size,size))
#title = 'Mut after RS. E:' + ('%.4f' % ent[0]) + '. '+ ('%.4f' % ent[1])
title = 'Mut. after RS. Fit:' + ('%.4f' % res) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])
plt_bar3d(output_folder + '/mut_after_' + str(run+1), diag.T, title, z_lim = 0.3)

#Save target
target = target.reshape((size,size))
plt_bar3d(output_folder + '/target', target.T, 'Target.', z_lim = 0.3)

f.close()


