"""
Random search with NO CMA

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
init_run = 0
runs = 30
seed = 21 #21
np.random.seed(seed) # Seed of seeds
seeds = ((np.random.rand(runs)*1e6)+100).astype(int)
running_index = [0,1,3,4,7]

# Variables
from config_experiment import *

## Output folder & variables
repetitions = Mu
output_folder = './output_mu/n_' + str(n_parts) #+ '_BA'

labels = ['Iter.', 'Fit', 'n1', 'n2', 'PHS']
labels_f =  "  {: <10} {: <10} {: <10} {: <10} {: <10}"

##################################################################
# Main flow
##################################################################

for run in  range(init_run, runs):

    my_index = running_index[run] #<== run
    np.random.seed(seeds[my_index ])

    condition = 'MU_' + str(Mu) + '_BA'
    
    name = output_folder + '/' + condition + '_' + str(run+1) + '.txt'
    
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

    # Generations
    for ite in range(max_ite + 1):
        start = time.perf_counter()
        improved = False
        if ite == 0:
            top_Mu = Mu_0
        else:
            top_Mu = Mu

        # Population
        for cont in range(top_Mu):
            new_decomp = rand_decomp(n_parts)
            new_circ = circuit(new_decomp)
            # No CMA
            if  True: #(ite+1) % 10 != 0: # Fraction of generations
                # No cma
                new_res = f_function(target, probs(evaluate(new_circ)))
                new_fit = new_res[0]
                new_ent, new_prob = new_res[1][0], new_res[1][1]
            else:
                # Constant CMA
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

                    # Optional CMA
                    #circ, fit, objs = cma_es(circ)
                    #ent = np.sum(objs[0])
                    #prob = objs[1]
                    
                    
        end = time.perf_counter()
        t = end - start

        tup = (ite, fit, n_resources(decomp)[0], n_resources(decomp)[1], n_PHS(circ), np.sum(ent), prob, t)
        print('%d. Fit: %.4f n1: %d n2: %d PHS: %d Ent: %.4f Prob: %.4f t: %.4f' % tup)
        f.write('%d\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f  \n' % tup )
    f.close()

    # Save circ
    name = output_folder + '/' + condition + '_' + str(run+1) + '_best.txt'
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
        
    name = output_folder + '/' + condition + '_' + str(run+1) + '.png'
        
    diag = diag.reshape((size,size))
    plt_bar3d(name, diag.T, title, z_lim = 0.3)

        



