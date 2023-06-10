"""
Genetic programming for polatization cicuit creation
Test:
for each in pop.generations[0]: print (each)
pop.best(2)[1:]
"""

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

########################################################
# Variables.
########################################################
init_run = 1
runs = 5
#output_folder = './output_GP/n_' + str(n_parts)
output_folder = './output_MO_1/n_' + str(n_parts)
    
labels = ['Iter.', 'Best', 'Avg', 'Worst', 'Calls', 'n1', 'n2', 'PHS', 't', 'Ent.', 'Prob.']
labels_f =  "  {: <10} {: <10} {: <10} {: <10}  {: <10}  {: <10}  {: <10}  {: <10}  {: <10}  {: <10}  {: <10}"

def file_data():
    global pop
    # Write to 4 s.f.
    idx = pop.best()[2]
    objs = pop.generations[-1][idx].objs
        
    string = ''
    for each in pop.performance():
        string += "%.4g" % each + '\t'
    string += "%s" % calls + '\t'
    string += "%s" % expr(pop.best()[0]).count('Sup.') + '\t'
    string += "%s" % expr(pop.best()[0]).count('Part.') + '\t'
    string += "%s" % n_PHS(pop.best()[0]) + '\t'
    string += "%.4g" % t + '\t'
    string += "%.4g" % objs[0] + '\t'
    string += "%.4g" % objs[1] + '\n'
    return string

def screen_data():
    global pop, cont, calls
    idx = pop.best()[2]
    objs = pop.generations[-1][idx].objs

    string_list = ["%3d" % (cont + 1)]
    string_list += ["%.4g" % each for each in pop.performance()]
    string_list += [str(calls)]
    string_list += ["%s" % expr(pop.best()[0]).count('Sup.')]
    string_list += ["%s" % expr(pop.best()[0]).count('Part.')]
    string_list += ["%s" % n_PHS(pop.best()[0]) ]
    string_list += ["%.4g" % t]
    string_list += ["%.4g" % objs[0]]
    string_list += ["%.4g" % objs[1]]     
   
    return string_list

def save_best_circ():
    global pop, run
    # Save best tree
    condition = 'GP'
    name = output_folder + '/' + condition + '__' + str(run+1) + '_best.txt'
    f = open(name, 'w')
    f.write(str(pop.best()[0]))
    f.close()
    return

def save_best_img():
    global output_folder, target, pop 
    res = pop.performance()[0]
    diag = probs( evaluate( pop.best()[0] ) )
    ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
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
    plt_bar3d(output_folder + '/run_' + str(run + 1), diag.T, title, z_lim = 0.3)

    t = target.reshape((size, size))
    plt_bar3d(output_folder + '/target', t.T, 'Target', z_lim = 0.3)
    return

########################################################
# Fitness function defined in functions_quantum_trees
########################################################
# This variable store calls to the fitness functions
calls = 0
def global_f_function(tree):
    global calls, target
    diag = probs(evaluate(tree))
    res = f_function(target, diag )
    calls += 1
    return res #[res, diag]

########################################################
# Random initialization and random seeds.
########################################################
if save:
    print('\n'*5 + 'SAVING RESULTS!' + '\n'*5)

seed = 51 ## Seed of seeds
np.random.seed(seed)
seeds = ((np.random.rand(runs)*1e6)+100).astype(np.int)
rep_seeds = ((np.random.rand(repetitions)*1e6)+100).astype(np.int)

for run in range (init_run, init_run+1):
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
        condition = 'GP'
        name = output_folder + '/' + condition + '__' + str(run+1) + '.txt'
        f = open(name, 'w')
        cad = ''
        for each in labels[1:]:
            cad += each + '\t'
        f.write( cad + '\n')
    
########################################################
# Create population.
########################################################
    pop = Population()
# Set the fitness function
    pop.set_fitness_function(global_f_function)

# First random generation of individuals of lenght the number of objects
    pop.first_generation()
    cont = -1; t = 0.
    if save:
        string = file_data()
        f.write( string)
    string_list = screen_data()
    print(labels_f.format(*string_list))

    for cont in range (0, max_ite):
        calls = 0
        start = time.perf_counter()
        
        # Recombine
        pop.recombine()
        
        # Mutate
        pop.mutate()
        # Reproduce
        pop.reproduce()
        
        # Selection
        pop.selection()
        end = time.perf_counter()
        t = end - start
        
        # Write to 4 s.f.
        if save:
            string = file_data()
            f.write( string)

        # Show
        if (cont == 0 or (cont + 1) % 1 == 0):
            string_list = screen_data()
            print(labels_f.format(*string_list))
            
    print()
    if save:
        f.close()
        # Save best circuit and image
        save_best_circ()
        save_best_img()
        
