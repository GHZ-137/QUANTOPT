"""
Create random quantum optics circuit for n parts.

Sources: Part.(Theta), superp.(Theta)
Tools: BS(Theta)(2), HWP(Theta)(1), Brewster(Theta)(1) => Not implemented

"""

from functions_basis import *
from functions_states import
from functions_quantum_trees import *
from functions_plot import *
from config_quantum import *
from config_experiment import *

# Variables
n_parts = 6

# Raturn a 2-state F according to config
def F():
    global F_func, F_mean, F_sigma, F_lim
    if F_func == 'Gaussian':
        res = np.random.normal(F_mean, F_sigma)
        np.clip(res, F_lim[0], F_lim[1])
    if F_func == 'Uniform':
        res = F_lim[0] + np.random.rand() * (F_lim[1] - F_lim[0])

    return res

# Raturn a 1-state angle according to config
def Theta():
    global single_func, single_mean, single_sigma, single_lim
    if single_func == 'Gaussian':
        res = np.random.normal(single_mean, single_sigma)
        np.clip(res, single_lim[0], single_lim[1])
    if single_func == 'Uniform':
        res = single_lim[0] + np.random.rand() * (single_lim[1] - single_lim[0])
    return res

# Raturn a HWP random angle according to config
def Theta_HWP():
    global HWP_func, HWP_mean, HWP_sigma, HWP_lim
    if HWP_func == 'Gaussian':
        res = np.random.normal(HWP_mean, HWP_sigma)
        np.clip(res, HWP_lim[0], HWP_lim[1])
    if HWP_func == 'Uniform':
        res = HWP_lim[0] + np.random.rand() * (HWP_lim[1] - HWP_lim[0])
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
    ##print (decomp)
    global p_extra_HWP
    extras = ''
    if np.random.rand() < p_extra_HWP:
        extras = ' HWP ' + str( np.round(Theta_HWP(), 2 ))

    # Terminal nodes
    if decomp == 1:
        return 'Sup. ' + str( np.round(Theta(), 2 )) +  extras
    if decomp == 2 :
        return 'Part. ' + str(np.round(F(), 2 )) + extras
        
    if len(decomp) == 2:
        res =  ['BS ' + str( np.round(Theta_BS(), 2 ) ) ,\
                  [ circuit(decomp[0]), circuit(decomp[1])]  ]

        if np.random.rand() < p_extra_HWP:
            extras = ' HWP ' + str(np.round(Theta_HWP(), 2 ))
            res.append( extras)
        return res

# Fitness function
def f_function(target, actual):
    #res = np.dot(target, actual)
    idx = list(np.where(target !=0)[0])
    a = 0
    b = 0
    for each in range(len(target)):
        if each in idx:
            a += actual[each]
        else:
            b += actual[each]
    res = a #/ b
    res = np.sum((target - actual)**2)
    return res

def print_probs(diag):
    print('P:' + '%.2f '* len(diag) % tuple(diag) )
    return

# Evaluate the same circuit with different noisy source states
def evaluate_multi(a):
    global target, repetitions
    #print('Solving circuit...')
    best_val = -float('inf')
    best = None
    fits = []
    for cont in range(repetitions):
        res = evaluate(a)
        diag = np.real(np.diag(res))
        fit = f_function(target, diag)
        if fit > best_val:
            best_val = fit
            best = diag
        fits.append(fit)
    return np.mean(fits)

#################################
# Main flow
np.random.seed(32)
#p_extra_HWP = 0.5 #5
#n = 5
repetitions = 10

# Target state
states = enum_states(n_parts)
target = np.zeros(2**n_parts)
target[0] = 0.25;
target[3] = 0.25;
target[-1] = 0.25;
target[-4] = 0.25;

# Circuit creation
decomp = rand_decomp(n_parts)

a = circuit(decomp)
print('Decomp. of '+ str(n_parts) + ':\n', decomp)
print('\nCircuit:\n', a )

print('Nodes:',n_nodes(a))
print('No term. nodes:',NT_nodes(a))
print('Depth:',depth(a), '\n')

# Do the magic
print('Solving circuit...')

# We evaluate the same circuit
# with different noisy source states

best_val = float('inf')
best = None
for cont2 in range(20):
    fits = []
    for cont in range(10):
        res = evaluate(a)
        diag = np.real(np.diag(res))
        fit = f_function(target, diag)
        if fit < best_val:
            best_val = fit
            best = diag
        fits.append(fit)
        #print('Fit: %.2f' % fit)
    print('Mean fit: %.2f' % np.mean(fits))

print('Best: %.2f' % np.mean(best_val))
print_probs(best)
diag = best
res = best_val

print(diag)
size = np.int(np.sqrt(len(diag)))

# Odd n_parts
if (n_parts % 2 != 0):
    diag2 = np.zeros((size+1)**2)
    diag2[: len(diag)] = diag
    diag = diag2
    size = np.int(np.sqrt(len(diag)))

diag = diag.reshape((size,size))
plt_grid('images/random_search', np.flip(diag, axis = 0), 'Random search. Fit:' + str(round(res,2)))

k = tree_indices(a)
tree_parts(tree_by_idx(a,k[0]))


