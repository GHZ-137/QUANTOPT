# Function to optimize a circuit parameters
# using CMA-ES
#
# Requires: pip3 install cmaes

########################################################
# Inclusions.
########################################################
from functions_quantum_trees import *
from config_GP import *
from cmaes import CMA

########################################################
# Variables.
########################################################
def cma_es(c):
    global init_sigma, bounds_orig, max_ite_cma

    prev_fit = f_function(target, probs(evaluate(c)))[0]
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
    
    for gen in range(max_ite_cma):
        evs = []
        best_val = -float('Inf')
        best_vars = []
        for each in range(my_cma.population_size):        
            c = c_orig[:]
            all_vars = tree_to_vars(c) # EXTRACT EXPLICITELY
            s = my_cma.ask()
        
            insert_eff_vars(s, all_vars)
            c = vars_to_tree(c, all_vars)
            # MO objective with Lambda
            ev = f_function( target, probs(evaluate(c)) )[0] #[1][0]
            # Entanglement objective
            #ev = f_function( target, probs(evaluate(c)) )[1][0]
            # Probability objective
            #ev = f_function( target, probs(evaluate(c)) )[1][1]
            
            if ev > best_val:
                best_val = ev
                best_vars = s
            evs.append( (s, -ev) )
            
        my_cma.tell(evs)
        all_vars = tree_to_vars(c)
        insert_eff_vars(best_vars, all_vars)
     
        new_tree = vars_to_tree(c, all_vars)
        fit, objs = f_function(target, probs(evaluate(new_tree)))
        objs = [ objs[0], objs[1] ] #np.sum(objs[0])
        
    return [new_tree, fit, objs]

"""
#Test
c=\
['PHS_88.18 BS_54.17_0.0', [['BS_11.42_0.0', [['BS_89.67_0.0', [['BS_41.71_2.64', [['PHS_85.3 Sup._42.2'], ['Sup._42.2']]], ['BS_49.0_0.0', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]], ['Sup._42.2']]]

res, prev, f = cma_es(c)
print(prev, f)
"""

