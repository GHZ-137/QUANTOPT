"""
Random search with optional CMA

"""
import time
from functions_basis import *
from functions_states import *
from functions_quantum_trees import *
from functions_cma_es import *
from functions_plot import *

from config_quantum import *
from config_GP import *

########################################################
# Functions
########################################################
def probs_target(circ):
    global pos_1, pos_2, pos_neg_1, pos_neg_2
    res = evaluate(circ)
    actual = np.real(np.diag(res))
    p_1 = np.sum( [actual[each] for each in pos_1] )
    p_2 = np.sum( [actual[each] for each in pos_2] )
    p_3 = np.sum( [actual[each] for each in pos_neg_1] )
    p_4 = np.sum( [actual[each] for each in pos_neg_2] )
    res = [p_1, p_2, p_3, p_4]
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

########################################################
# Variables
########################################################
init_run = 1
runs = 5
seed = 21 #21
np.random.seed(seed) # Seed of seeds
seeds = ((np.random.rand(30)*1e6)+100).astype(int)
running_index = [0,2,3,4,7,8]

# Variables
from config_experiment import *

## Output folder & variables
repetitions = Mu
output_folder = './output_pg+cma/n_' + str(n_parts)

labels = ['Iter.', 'Fit', 'n1', 'n2', 'PHS']
labels_f =  "  {: <10} {: <10} {: <10} {: <10} {: <10}"

##################################################################
# Main flow
##################################################################

for run in range(init_run, runs):
    run_idx = running_index[run]
    np.random.seed(seeds[run_idx])
    
    condition = 'Random_search_' + str(run + 1)
    name = output_folder + '/' + condition + '.txt'
    f = open(name, 'w')
    cad = ''
    for each in labels[:]:
        cad += each + '\t'
    f.write( cad + '\n')

    # Initial circuit: random search
    decomp = rand_decomp(n_parts)
    print('Decomp. of '+ str(n_parts) + ':\n', decomp)
    circ = circuit(decomp)
    res = f_function(target, probs(evaluate(circ)))
    fit = res[0]
    ent, prob = res[1][0], res[1][1]

    improved = False
    cont_improved = False
    
    for ite in range(max_ite + 1):
        start = time.perf_counter()

        my_Mu = Mu_0
        if ite > 0:
            my_Mu = Mu

        print(improved)
        for cont in range(my_Mu):
            new_decomp = rand_decomp(n_parts)
            new_circ = circuit(new_decomp)
            
            new_res = f_function(target, probs(evaluate(new_circ)))
            new_fit = new_res[0]
            new_ent, new_prob = new_res[1][0], new_res[1][1]
            print(cont)
            # Constant, CMA
            if improved or ite % 5 == 0: # 1/5 of generations
                new_circ, new_fit, objs = cma_es(new_circ)
                new_ent, new_prob = np.sum(objs[0]), objs[1]
            
            if optimization == 'Minimize':
                if new_fit < fit:
                    decomp = new_decomp
                    circ = new_circ
                    fit = new_fit
                    ent, prob = new_ent, new_prob         
            else:
                if new_fit > fit:
                    decomp = new_decomp
                    circ = new_circ
                    fit = new_fit
                    ent, prob = new_ent, new_prob

                    # If improvement, keep cont_improved -> improved
                    circ, fit, objs = cma_es(circ)
                    ent = np.sum(objs[0])
                    prob = objs[1]
                    cont_improved = True
        improved = cont_improved and True
        cont_improved = False
                    
        end = time.perf_counter()
        t = end - start

        tup = (ite, fit, n_resources(decomp)[0], n_resources(decomp)[1], n_PHS(circ), np.sum(ent), prob, t)
        print('%d. Fit: %.4f n1: %d n2: %d PHS: %d Ent: %.4f Prob: %.4f t: %.4f' % tup)
        f.write('%d\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f  \n' % tup )
    f.close()

    # Save circ
    name = output_folder + '/' + condition + '_best.txt'
    f = open(name, 'w')
    f.write(str(circ))
    f.close()

    #import ast
    #ast.literal_eval(str(circ))

    # Save image
    diag = probs( evaluate( circ ) )
    ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
    title = 'Random search. Fit:' + ('%.4f' % fit) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])
    size = int(np.sqrt(len(diag)))
    # If Odd n_parts
    if (n_parts % 2 != 0):
        diag2 = np.zeros((size+1)**2)
        diag2[: len(diag)] = diag
        diag = diag2
        size = int(np.sqrt(len(diag)))
    diag = diag.reshape((size,size))
    plt_bar3d(output_folder + '/Rnd_' + str(run + 1), diag.T, title, z_lim = 0.3)

        



