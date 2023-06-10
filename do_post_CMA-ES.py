# CMA-ES to optimize condition results.
#
#
# Indicate n_parts in config_experiment
# Paste the circuit into variable 'c'
#
# Requires: pip3 install CMA

########################################################
# Inclusions.
########################################################
import warnings; warnings.filterwarnings('ignore')
import time, pickle, math, numpy as np

from classes_quantum_GP import *
from functions_quantum_trees import *
from functions_plot import *

from config_quantum import *
from config_experiment import *
from config_GP import *
from cmaes import CMA

########################################################
# Variables.
########################################################
init_run = 0
runs = 5
init_sigma = 1.0
bounds_orig = [0.1, 90] #[1, 89.]

condition = 'MU_2000_BA'
#condition = 'MU_2000_GP_0.5_200_SOFT_2'

output_folder = './output_cma/n_' + str(n_parts)
input_folder = './output_mu/n_' + str(n_parts)
labels = ['Iter.', 'Best', 't', 'Ent.', 'Prob.']
labels_f =  "  {: <10} {: <10}{: <10}  {: <10}  {: <10}"

def file_data(t, fit, objs):
    # Write to 4 s.f.
   
    string = ''
    string += "%.4g" % fit + '\t'
    string += "%.4g" % t + '\t'
    string += "%.4g" % objs[0] + '\t'
    string += "%.4g" % objs[1] + '\n'
    return string

def screen_data(cont, t, fit, objs):

    string_list = ["%3d" % (cont + 1)]
    string_list += ["%.4g" %fit]
    string_list += ["%.4g" % t]
    string_list += ["%.4g" % objs[0]]
    string_list += ["%.4g" % objs[1]]     
   
    return string_list

def save_best_circ():
    global pop, run, condition
    # Save best tree
    name = output_folder + '/' + condition + '__' + str(run+1) + '_best.txt'
    f = open(name, 'w')
    f.write( str(actual) )
    f.close()
    return

def save_best_img():
    global output_folder, target, condition
    diag = probs( evaluate( actual ) )
    ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
    res = ent[0] + ent[1]
    print('E:')
    for each in ent: print('%.4f\t' % each)
    title = 'GP. Fit:' + ('%.4f' % res) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])
    
    size = int(np.sqrt(len(diag)))
    # If Odd n_parts
    if (n_parts % 2 != 0):
        diag2 = np.zeros((size+1)**2)
        diag2[: len(diag)] = diag
        diag = diag2
        target2 = np.zeros((size+1)**2)
        target2[: len(target)] = target
        target = target2
            
        size = np.int(np.sqrt(len(diag)))

    diag = diag.reshape((size,size))
    plt_bar3d(output_folder + '/' + condition +'_run_' + str(run + 1) + '.png', diag.T, title, z_lim = 0.3)

    t = target.reshape((size, size))
    plt_bar3d(output_folder + '/target', t.T, 'Target', z_lim = 0.3)
    return

########################################################
# Random initialization and random seeds.
########################################################
if save:
    print('\n'*5 + 'SAVING RESULTS!' + '\n'*5)

seed = 20 ## Seed of seeds
np.random.seed(seed)
seeds = ((np.random.rand(runs)*1e6)+100).astype(np.int)

for run in range (init_run, init_run+5):
# Reset calls in each run
    calls = 0
    print('\n Run:', str(run+1))
    print(labels_f.format(*(labels)))
    
########################################################
# Associating random seeds to runs allows
# to continue interrupted simulations.
########################################################
    run_seed = seeds[run]
    np.random.seed(run_seed)
    
# Outuput file
    if save:
        name = output_folder + '/' + condition + '_' + str(run+1) + '.txt'
        f = open(name, 'w')
        cad = ''
        for each in labels[1:]:
            cad += each + '\t'
        f.write( cad + '\n')
    
########################################################
# CMA-ES
########################################################
    # Load circuit
    name = input_folder + '/' + condition + '_' + str(run+1) + '_best.txt'
    f2 = open(name, 'r')
    c = eval( f2.readline() )
    f2.close()
    
    # Secure original circuit
    c_orig = c[:]
    c = c_orig[:]
    orig_vars = tree_to_vars(c)
    
    # Initial values
    vars_ = extract_eff_vars(orig_vars)
    n_vars = len(vars_)
    bounds = np.array( [bounds_orig]* n_vars )
    init_vals = np.array( vars_ ) 
    
    # CMAES
    my_cma = CMA(init_vals, init_sigma)
    my_cma.set_bounds(bounds)
    
    for gen in range(max_ite):
        start = time.perf_counter() 
        evs = []
        best_val = -float('Inf')
        best_vars = []
        for each in range(my_cma.population_size):        
            c = c_orig[:]
            all_vars = tree_to_vars(c) # EXTRACT EXPLICITELY
            s = my_cma.ask()
        
            insert_eff_vars(s, all_vars)
            c = vars_to_tree(c, all_vars)
            ev = f_function( target, probs(evaluate(c)) )[0] #[1][0]
            if ev > best_val:
                best_val = ev
                best_vars = s
            evs.append( (s, -ev) )
            
        my_cma.tell(evs)
        all_vars = tree_to_vars(c)
        insert_eff_vars(best_vars, all_vars)
     
        actual = vars_to_tree(c, all_vars)
        fit, objs = f_function(target, probs(evaluate(actual)))
        objs = np.sum(objs[0]), objs[1]
        ##print('%d, %.2f' % (gen, fit ))
        
        # End
        end = time.perf_counter()
        t = end - start
        
        # Write
        if save:
            string = file_data(t, fit, objs)
            f.write( string)

        # Show
        if True:
            string_list = screen_data(gen, t, fit, objs)
            print(labels_f.format(*string_list))
        #print()
    
    if save:
        f.close()
        # Save best circuit and image
        save_best_circ()
        save_best_img()
        
