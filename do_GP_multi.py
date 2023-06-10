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
import concurrent.futures

if __name__ == '__main__':
    #TABULATE
    
########################################################
# Variables.
########################################################
    init_run =0
    runs = 5
    running_index = [0,2,3,4,7,8]
    #running_index = [2,0,3,4,7,8]
    
    output_folder = './output_nsga/n_' + str(n_parts) + '_test'

    labels = ['Iter.', 'Best', 'Avg', 'Worst', 'Calls', 'n1', 'n2', 'PHS', 'Ent.', 'Prob.']
    labels_f =  "  {: <10} {: <10} {: <10} {: <10}  {: <10}  {: <10}  {: <10}  {: <10}  {: <10}  {: <10}"

    if do_NSGA:
        labels += ['size_F1', 't']
        labels_f += " {: <10} {: <10}"
    else:
        labels += ['t']
        labels_f += " {: <10}"      
        
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
        string += "%.4g" % objs[0] + '\t'
        string += "%.4g" % objs[1] + '\t'

        if do_NSGA:
            n = pop.f1_size #len( np.where(pop.front_arr == 0)[0] )
            string += "%d" %n + '\t'
            
        string += "%.4g" % t + '\n'
        
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
        string_list += ["%.4g" % objs[0]]
        string_list += ["%.4g" % objs[1]]

        if do_NSGA:
            n = pop.f1_size #len(np.where(pop.front_arr == 0)[0])
            
            string_list += ["%d" %n]
            
        string_list += ["%.4g" % t]
   
        return string_list

    def save_best_circ():
        global pop, run, condition
        # Save best tree
        #condition = 'GP'
        name = output_folder + '/' + condition + '_' + str(run+1) + '_best.txt'
        f = open(name, 'w')
        f.write(str(pop.best()[0]))
        f.close()
        return

    def save_best_img():
        global output_folder, target, pop, run, condition 
        res = pop.performance()[0]
        diag = probs( evaluate( pop.best()[0] ) )
        ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
        #res = ent[0] + ent[1]
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


        name = output_folder + '/' + condition + '_' + str(run+1) + '.png'
       
        diag = diag.reshape((size,size))        
        plt_bar3d(name , diag.T, title, z_lim = 0.3)

        t = target.reshape((size, size))
        plt_bar3d(output_folder + '/target', t.T, 'Target', z_lim = 0.3)
        return

########################################################
# Multiprocess
########################################################
    def shared_evaluate(inds):
        global global_f_function
        fit = []
        obj = []
        for ind in inds:
            val = global_f_function(ind.vars)
            fit.append( val[0] )
            obj.append( val[1] )
        return [fit, obj]

    def shared_mutate(inds):
        res = []
        for ind in inds:
            ind.mutate()
            res.append( ind )
        return res

    def shared_recombine(n_pairs):
        res = []
        for cont in range(n_pairs):
            # Parents must have same number of parts in subtrees
            parent_coincide = False
            while not(parent_coincide):

                # 2 parents by tournament from previous generation.
                parents = self.tournament(prev = -1, n = 2)
    
                # Parents need cloning and duplicating
                p1_1 = deepcopy(parents[0])
                p1_2 = deepcopy(parents[0])
                p2 = deepcopy(parents[1])

                # Choose random idx from parent 1 and locate in parent 2
                # Get random node number
                ln = len(p1_1.tree_idx)
                val = ln 
                idx1 = np.random.choice( range(1, val) )
            
                # How many parts are in the node?
                n1 = n_sources( tree_by_idx(p1_1.tree, p1_1.tree_idx[idx1]) )
            
                n_equals = False
                idx2 = 0
                while not n_equals and idx2 < len(range(len(p2.tree_idx) )):
                    n2 = n_sources( tree_by_idx(p2.tree, p2.tree_idx[idx2]) )
                    #print (n1, n2)
                    if n2 == n1:
                        n_equals = True
                    idx2 += 1
                idx2 -= 1

                if n_equals:
                    parent_coincide = True
                        
            # Update and store
            p1_1.update(); p2.update()
            res.extend([p1_1, p2])
            
        return res

    def pool_evaluate(inds):
        res = do_pool(inds, 'evaluate')
        return res

    def pool_mutate(inds):
        res = do_pool(inds, 'mutate')
        return res
    
    def pool_recombine(inds):
        res = do_pool(inds, 'recombine')
        return res

    def do_pool(inds, arg): #, f_function, n_procs):
        global n_procs
        # Assign
        jobs = []
        l = len(inds)
        size = l // n_procs
        res = []
        for i in range(n_procs):
            my_inds = inds[ size * i : size * (i+1)]
            if i == (n_procs - 1):
                my_inds = inds[ size * i : ]  
            jobs.append( my_inds)
            
        # Execute and collect
        res = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = executor.map( shared_evaluate, jobs)
            
            if arg == 'evaluate':
                pool = executor.map( shared_evaluate, jobs)
            if arg == 'mutate':
                pool = executor.map( shared_mutate, jobs)
            if arg == 'recombine':
                pool = executor.map( shared_recombine, jobs)
              
        if arg == 'evaluate':
            # Separate fit and objs
            fit = []; obj = []
            for r in pool:
                fit.extend(r[0])
                obj.extend(r[1])
            res = [fit, obj]
             
        else:
            res = []
            for r in pool:
                res.extend(r)
           
        return res

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
    print('Multi:', use_multi)
    
    seed = 21 ## Seed of seeds
    np.random.seed(seed)
    seeds = ((np.random.rand(100)*1e6)+100).astype(np.int)
    #rep_seeds = ((np.random.rand(repetitions)*1e6)+100).astype(np.int)

    for run in range(init_run, runs): #range (init_run, runs): #init_run+1):
    # Reset calls in each run
        calls = 0
        print('\n Run:', str(run+1))
        print(labels_f.format(*(labels)))
    
########################################################
# Associating random seeds to runs allows
# to continue interrupted simulations.
########################################################
        my_index = running_index[run] #<== run
        np.random.seed(seeds[my_index ])
    
    # Outuput file
        if save:
            
            ope = 'SUBTREE'
            if soft_mutation:
                ope = 'SOFT_' + str(soft_sigma)
            condition = 'MU_' + str(Mu) + '_GP_' + str(p_c) + '_' + str(Gamma)

            if freq_cma != 0:
                condition = 'CMA_' + str(frac_ind_cma) + '_'  + str(max_ite_cma) + '_' + condition

            condition = condition + '_' + ope
    
            name = output_folder + '/' + condition + '_' + str(run+1) + '.txt'
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
        pop.multi_ev_f = pool_evaluate
        pop.multi_mut_f = pool_mutate

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
            pop.selection(run, cont+1, output_folder)
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
        
