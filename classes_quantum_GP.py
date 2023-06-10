"""
GP classes:
    Individual
    Population

"""
########################################################
# Inclusions.
########################################################
import numpy as np
import pickle

from functions_quantum_trees import *
from functions_basis import *
from functions_states import *
from functions_cma_es import *
from functions_NSGA import *

from config_quantum import *
from config_experiment import *
from config_GP import *

########################################################
# Each individual is a random tree with
# random initialization (using global n_parts)
# random sub-tree mutation
########################################################
class Individual:
    def __init__(self):
        global n_parts
        decomp = rand_decomp(n_parts)
        self.tree = circuit(decomp)
        self.vars = self.tree
        self.tree_idx = tree_idx( self.tree)
        return

    def mutate(self):
        # PG mutation
        do_soft = False
        if soft_mutation and np.random.rand() < 1 : do_soft = True
        if not(do_soft):
            # Get random node different from root and terminal node == 1
            term = True
            while term:
                node_idx = self.tree_idx[ np.random.choice( range( 1, len(self.tree_idx) ))] #0
                term = node_type(  tree_by_idx( self.tree, node_idx) ) == 'T'
                term = term and n_sources( tree_by_idx(self.tree, node_idx) ) == 1
            #print(term)
            
            # How many parts are in the node?
            n_sub = n_sources( tree_by_idx(self.tree, node_idx) )
            #print(n_sub)
            # Substitute the node in position node_idx by a random tree with same number of parts
            decomp = rand_decomp(n_sub)
            new_node = circuit(decomp)

            substitute(self.tree, node_idx, new_node)

            # Update tree indices
            self.tree_idx = tree_idx( self.tree)
            self.vars = self.tree
            
        # Soft mutation of the numeric parameters in a node
        else:
            all_vars = tree_to_vars(self.tree)
            sel = [ np.random.choice( range(len(all_vars)) ) ]
            if soft_all:
                sel = range(len(all_vars))
            for idx in sel:
                if all_vars[idx][0] == 'BS':
                    all_vars[idx][1] += np.random.normal(0, soft_sigma) #Theta_BS()
                    
                    if p_BS_PHS_change != 0: # PHS_BS not allowed
                        all_vars[idx][2] += np.random.normal(0, soft_sigma)
                        
                    all_vars[idx][1] = np.clip(all_vars[idx][1], Theta_lim[0], Theta_lim[1])
                    all_vars[idx][2] = np.clip(all_vars[idx][2], Phi_lim[0], Phi_lim[1])
                if all_vars[idx][0] == 'PHS':
                    all_vars[idx][1] += np.random.normal(0, soft_sigma)
                    all_vars[idx][1] = np.clip(all_vars[idx][1], Theta_lim[0], Theta_lim[1])
                    
            self.tree = vars_to_tree(self.tree, all_vars)
            self.update()
        return
    
    def update(self):
        self.tree_idx = tree_idx( self.tree)
        self.vars = self.tree
        return
    
"""
# Tests
n_parts = 5
np.random.seed(20)
a = Individual()
print( a.tree)
print(n_sources(a.tree), n_nodes(a.tree), depth(a.tree))

a.mutate()
print( a.tree)
print(n_sources(a.tree), n_nodes(a.tree), depth(a.tree))
"""

########################################################
# Population class.
########################################################
class Population:
    def __init__(self):
        # Constructor method.
        # generations is initially an empty list
        self.generations = []
        # List of ind. idx ordered by fitness. First(0) -> Best 
        self.fit_order = []
        return

    def set_fitness_function(self, p_function):
        # Set fitness function pointer
        self.fitness_function = p_function
        return

    def first_generation(self):
        global Mu, Mu_0
        generation = []
        for cont in range(Mu_0):
            generation.append( Individual() )

        self.generations.append(generation)
        self.evaluate()

        if do_NSGA:
            # Compute fronts, its indices and crowd_d
            new_Mu = self.generations[-1] 
            objs_l = []
            for each in new_Mu:
                objs_l.append( each.objs)# [::-1]) #Invert    <<<<

            self.front_arr, self.crowd_arr, _, _, _ = front_crowd_arr(objs_l) # <<< Update

            my_fronts = fronts(objs_l)
            self.f1_size = len(my_fronts[0])

        return

    def evaluate(self, n_gen=-1):
        # Store fitness value of each ind. in value.
        # Applies to n_gen number (default -1, present generation)
        global optimization, freq_cma
        
        # Evaluate individuals
        
        # cma between generations?
        if freq_cma > 0:
            k = len(self.generations) % (freq_cma)
            if k == 0:
                for ind in self.generations[n_gen]:
                    ind.vars(ind.vars)[0]
                    ind.tree = ind.vars

        if use_multi:
            inds = self.generations[n_gen]
            res = self.multi_ev_f(inds)
            
            for cont in range(len( self.generations[n_gen] )):
                self.generations[n_gen][cont].value = res[0][cont]
                # Sum the entanglement alternatives
                objs = [ np.sum(res[1][cont][0]), res[1][cont][1] ]
                self.generations[n_gen][cont].objs = objs
                
        else:
            for ind in self.generations[n_gen]:
                #print(tree_parts(ind.vars))
                res = self.fitness_function(ind.vars)
                
                ind.value = res[0]
                ind.objs = [np.sum(res[1][0]), res[1][1]]
  
        fits = [ind.value for ind in self.generations[n_gen]]
        if optimization == 'Minimize':
            f_order = np.argsort(fits)
        elif optimization == 'Maximize':
            f_order = np.argsort(fits)[::-1]
        self.fit_order.append(f_order)
        return

    def best(self, n_gen=-1):
        # Best individual in a generation.
        # Return vars. (tree), fitness and index
        
        idx = self.fit_order[n_gen][0]
        best_ind = self.generations[n_gen][idx].vars
        best_val = self.generations[n_gen][idx].value
        return [best_ind, best_val, idx]

    def worst(self, n_gen=-1):
        # worst individual in a generation.
        # Return vars. (tree), fitness and index
        
        idx = self.fit_order[n_gen][-1]
        w_ind = self.generations[n_gen][idx].vars
        w_val = self.generations[n_gen][idx].value   
        return [w_ind, w_val, idx]

    def average(self, n_gen=-1):
        # Average ind. fitness in n_gen number
        avg = 0  
        for ind in self.generations[n_gen]:
            avg += ind.value
        avg /= len(self.generations[n_gen])   
        return avg

    def tournament(self, prev = -1, n = 2):
        # List of n individuals by n random tournments of size Gamma.
        # 'prev' indicates to use last (-1, default) or previous to last generation (-2).
        # 'prev = -2' should be used in sequential recombination + mutation + reproduction
        
        global Gamma, do_NSGA

        if do_NSGA:
            res = []
            for cont in range(n):
                # Binary crowded comparison            
                rnd_idx = np.round((np.random.rand(2)) * (len(self.generations[prev])-1)).astype(np.int)
                ind_1 = self.generations[prev][rnd_idx[0]]
                ind_2 = self.generations[prev][rnd_idx[1]]
                f_1 = self.front_arr[rnd_idx[0]]
                f_2 = self.front_arr[rnd_idx[1]]
                c_1 = self.crowd_arr[rnd_idx[0]]
                c_2 = self.crowd_arr[rnd_idx[1]]
            
                if f_1 < f_2:
                    res.append(ind_1)
                if f_2 < f_1:
                    res.append(ind_2)
                    
                if f_1 == f_2:
                    if c_1 > c_2:
                        res.append(ind_1)
                    else:
                        res.append(ind_2)
            
        else:
            res = []
            for cont in range(n):
                # Random indices
                rnd_idx = np.round((np.random.rand(Gamma)) * (len(self.generations[prev])-1)).astype(np.int)
                # Fitness order of random indices and best of them
                my_fit_order = self.fit_order[prev + 1][rnd_idx]
                best_of_rnd_idx = np.where(my_fit_order == np.min(my_fit_order))[0][0]
        
                ##print(rnd_idx, my_fit_order, best_of_rnd_idx)
                ##print( rnd_idx[best_of_rnd_idx ])
                sel = rnd_idx[best_of_rnd_idx ]
                best_ind = self.generations[prev][sel]
            
                res.append( best_ind )
        return res
    
    def recombine(self):
        # Get (Mu * p_c) children by recombining two random tournament parents
        # Recombination should respect:
        #  n_parts of sub-trees
        # A random sub-tree is chosen from parent 1 (excluding parent tree)
        # and the sub-tree with the same number of resources is located in parent 2.
        # Both sub-trees are swapped.
    
        global Mu, Lambda_size, p_c, elitism

        # If elitism, add previous best to children. Except if NSGA!
        children = []
        if elitism == 'Yes' and not do_NSGA:
            children = [ self.generations[-1][self.best(-1)[2]] ]

        # Compute front and crowding arrays
        if do_NSGA:
            # Truncate first front < 0.95 * Mu:
            objs_l = [ind.objs for ind in self.generations[-1] ] # Invert [::-1]   <<<
            self.front_arr, self.crowd_arr, _, _, _ = front_crowd_arr(objs_l, int(round(1 * Mu)) )

     
        for cont in range( np.floor(Lambda_size * p_c / 2.).astype(int) ):
            # Parents must have same number of parts in subtrees
            parent_coincide = False
            while not(parent_coincide):

                # 2 parents by tournament from previous generation.
                parents = self.tournament(prev = -1, n = 2)

                # IF NSGA -> biased tournament by crowd comparison < < <
                """
                if do_NSGA:
                    idx1 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                    idx2 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                    parents = [self.generations[-1][idx1], self.generations[-1][idx2]]
                """
                
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
                
                #shuffled_idx = list(range(0, len(p2.tree_idx)))
                #np.random.shuffle(shuffled_idx)
                while not n_equals and idx2 < len(p2.tree_idx):
                    #sel_idx = shuffled_idx[idx2]
                    n2 = n_sources( tree_by_idx(p2.tree, p2.tree_idx[idx2]) )
                    #print (n1, n2)
                    nterm1 =   node_type( tree_by_idx( p1_1.tree, p1_1.tree_idx[idx1])  ) == 'NT'
                    nterm2 =   node_type( tree_by_idx( p2.tree, p2.tree_idx[idx2])  ) == 'NT'

                    # One of two nodes, at least, should not be terminal
                    if n2 == n1 and (nterm1 or nterm2):
                        n_equals = True
                    idx2 += 1
                    
                idx2 -= 1
                #idx2 = shuffled_idx[idx2]
        
                if n_equals:
                    parent_coincide = True
                
            # Crossing
            substitute(p1_1.tree, p1_1.tree_idx[idx1], tree_by_idx( p2.tree, p2.tree_idx[idx2] ))
            substitute(p2.tree, p2.tree_idx[idx2], tree_by_idx( p1_2.tree, p1_2.tree_idx[idx1] ))
            p1_1.update(); p2.update()

            children.extend([p1_1, p2])
        self.generations.append( children )
        return

    def mutate(self):
        # Get (Mu * p_m) children by mutation.
        # As we have stored previous recombination results,
        # parents should be selected from previous to last generation (n = -2)
        
        global Mu, Lambda_size, p_m
        children = []

        if use_multi:
            parents = []
            for cont in range( np.round(Mu * p_m ).astype(int) ):
                parents.append( self.tournament(prev = -2, n = 1)[0] )
                
            res = self.multi_mut_f(parents)
            children = res
            
        else:
            for cont in range( np.floor(Lambda_size * p_m ).astype(int) -1 ):
                parent = self.tournament(prev = -2, n = 1)[0]
                # IF NSGA -> biased tournament by crowd comparison < < <
                """
                # unless NSGA -> random selection << <
                if do_NSGA:
                    idx1 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                    parent = self.generations[-1][idx1]
                """

                parent_1 = deepcopy(parent)
                parent_1.mutate()
     
                children.append(parent_1)
        self.generations[-1].extend( children )
        return

    def reproduce(self):
        # Get (Mu * p_r) children by selection
        # from previous to last generation (n = -2)
        global Mu, p_r, freq_cma, optimization

        children = []
        for cont in range( Mu - len(self.generations[-1]) ):

            # IF NSGA -> biased tournament by crowd comparison < < <
            """
            if do_NSGA:
                idx1 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                parent = self.generations[-1][idx1]
            """     
            parent = self.tournament(prev = -2, n = 1)[0]
            children.append(parent)
        self.generations[-1].extend( children )

        # Finally, do not forget evaluating present generation! < < <
        self.evaluate()
        
        # cma after improvement?
        best_prev = self.best(n_gen=-2)[1]
        best = self.best(n_gen=-1)[1]
        
        if (freq_cma == -1 ) and \
        ((optimization == 'Maximize' and best > best_prev)\
           or (optimization == 'Minimize' and best < best_prev)):
            # Inds. to apply cma
            #
            my_fit_order = self.fit_order[-1]
            l_inds = [self.generations[-1][each] for each in my_fit_order][: int(np.round(Mu * frac_ind_cma))]
            print(len(l_inds))
            #l_inds = self.generations[-1]

            
            for ind in l_inds: #[self.best(-1)[2]]]: #If only the best
                res = cma_es(ind.vars)
                ind.vars = res[0]
                ind.tree = ind.vars
                
                #res = self.fitness_function(ind.vars)
                ind.value = res[1] #res[0]
                ind.objs = [np.sum(res[2][0]), res[2][1]] #[np.sum(res[1][0]), res[1][1]]
            self.evaluate()
             
        return

    def selection(self, run =0, cont = 0, out_folder = ''):
        # Generational substitution do nothing
        pass
    
        # ONLY IF NSGA
        if do_NSGA: 
            # Join Mu and Lambda
            new_Mu = self.generations[-1] + self.generations[-2]

            """
            # Compute again fronts, its indices and crowd_d            
            objs_l = []
            for each in new_Mu:
                objs_l.append( each.objs) #::-1])   #Invert  <<<<<<

            ###self.front_arr, self.crowd_arr = front_crowd_arr(objs_l) # < < Update
  
            my_fronts = fronts(objs_l)
            self.f1_size = len(my_fronts[0])
            
            my_fronts_idx = [front_idx(each, objs_l) for each in my_fronts]
            my_fronts_crowd = [crowd_dist(each) for each in my_fronts]
            """

            # Truncate first front < 0.95 * Mu:
            objs_l = [ind.objs for ind in new_Mu]
            self.front_arr, self.crowd_arr, my_fronts_idx, my_fronts_crowd, my_fronts =\
                            front_crowd_arr(objs_l, int(round(1 * Mu)) )

            self.f1_size = len(my_fronts_idx[0])
         
            # CMA
            if do_NSGA_CMA:
                print("CMA-ES:", len(my_fronts_idx[0]))
                for idx in my_fronts_idx[0]:
                    ind = new_Mu[idx]
                    res = cma_es(ind.vars)
                    ind.vars = res[0]
                    ind.tree = ind.vars
                
                    # Entanglement objective [1]
                    # Probability objective [2][1]
                    ind.value = res[1]
                    ind.objs = [np.sum(res[2][0]), res[2][1]] #[np.sum(res[1][0]), res[1][1]]  <<<<
                self.evaluate()
                
            # Save
            if True: ###cont == 1 or cont == 100 or cont%10 == 0:
                f3 = open(out_folder + '/' + str(run+1)+ '_pareto_' + str(cont) , "wb")
                pickle.dump(my_fronts[0], f3)
                f3.close()
                
                plt.clf()
                plt.ylim(0,2) # Pareto_graph_limit)
                plt.xlim(0,1) # Pareto_graph_limit);
                for cont2 in range(1)[::-1]:
                    p = np.array(my_fronts[cont2])
                    plt.scatter(p[:,1], p[:,0])
                    plt.xlabel("P(E)")
                    plt.ylabel("E")
                    plt.title('Pareto front. N=' + str(n_parts) + '. Gen:' + str(cont))
                    plt.savefig(out_folder + '/' + str(run+1)+ '_pareto_' + str(cont) + '.png') #

            # Assign fronts to new_Lambda
            new_Lambda = []
            #if elitism:
                #new_Lambda = [ self.generations[-2][self.best(-2)[2]] ]
            
            cont = 0
            ##print( len(my_fronts) ) 
            if len(my_fronts)> cont:
                while (len(new_Lambda) + len(my_fronts[cont])) <= Mu :  
                    to_add = [new_Mu[each] for each in my_fronts_idx[cont] ]
                
                    new_Lambda.extend( to_add )
                    ##print( len(new_Lambda) )
                    cont += 1
                   
            # Truncate last front by decreasing crowding
            dist = crowd_dist(objs_l[cont])
            dist_idx = np.argsort(dist)[::-1]
            n_diff = Mu - len(new_Lambda)
            ##print(n_diff)

            to_add = [new_Mu[each] for each in my_fronts_idx[cont]]
            
            to_add2 = [to_add[each] for each in dist_idx[:n_diff]]
            new_Lambda.extend(to_add2)

            ##print(len(new_Lambda))
            
            fits = [ind.value for ind in new_Lambda ]
            new_fit_order = np.argsort(fits)[::-1]
                    
            self.generations[-1] = new_Lambda
            self.fit_order[-1] = new_fit_order
            
        return

    def performance(self, n_gen = -1):
        # Obtain best / average / worst performance in n_gen number
        return [self.best(n_gen)[1], self.average(n_gen), self.worst(n_gen)[1]]

