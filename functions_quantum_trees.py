# Tree functions.
# Each time a tree is created or modified,
# its node index list must be updated.
# Ex:
#   a = tree(8, 'grow');
#   a_idx = tree_idx(a);

import numpy as np
from copy import *
from itertools import *

from functions_states import *
from config_quantum import *
from config_GP import *
from config_experiment import *

########################################################
# Circuit creation
#
########################################################
# Return a 2-state F according to config
def F():
    global F_func, F_mean, F_sigma, F_lim
    if F_func == 'Gaussian':
        res = np.random.normal(F_mean, F_sigma)
        np.clip(res, F_lim[0], F_lim[1])
    if F_func == 'Uniform':
        res = F_lim[0] + np.random.rand() * (F_lim[1] - F_lim[0])

    return res

# Raturn a 1-state Theta value according to config
def Theta_sup():
    global single_func, single_mean, single_sigma, single_lim
    if single_func == 'Gaussian':
        res = np.random.normal(single_mean, single_sigma)
        np.clip(res, single_lim[0], single_lim[1])
    if single_func == 'Uniform':
        res = single_lim[0] + np.random.rand() * (single_lim[1] - single_lim[0])
    return res

# Raturn BS random Theta and Phi according to config
def Theta_BS():
    global Theta_func, Theta_mean, Theta_sigma, Theta_lim
    if Theta_func == 'Gaussian':
        res = np.random.normal(Theta_mean, Theta_sigma)
        np.clip(res, Theta_lim[0], Theta_lim[1])
    if Theta_func == 'Uniform':
        res = Theta_lim[0] + np.random.rand() * (Theta_lim[1] - Theta_lim[0])

    #disc_vals = [0., 22.5, 45., 67.5, 90.]
    #vals = np.copy(disc_vals)
    #np.random.shuffle(vals)
    #res = vals[0]
    
    return res

def Phi_BS():
    global Phi_func, Phi_mean, Phi_sigma, Phi_lim
    if Phi_func == 'Gaussian':
        res = np.random.normal(Phi_mean, Phi_sigma)
        np.clip(res, Phi_lim[0], Phi_lim[1])
    if Phi_func == 'Uniform':
        res = Phi_lim[0] + np.random.rand() * (Phi_lim[1] - Phi_lim[0])

    #disc_vals = [0., 22.5, 45., 67.5, 90.]
    #vals = np.copy(disc_vals)
    #np.random.shuffle(vals)
    #res = vals[0]
    
    return res


# Random integer decompositions of n parts into single and double source states
def rand_decomp(n):
    if n == 1:
        res = [1]
    elif n == 2:
        a = [ [ 1,1 ], 2 ]
        ##a = [ [ 1,1 ], [ 1,1 ] ] #Fixed circuit
        
        # change probability!
        res = [ a[np.random.choice([0,1])] ]
    else:
        cont1 = n-1
        cont2 = 1
        res = []
        while (cont1 >= cont2): #AQUI
            res.append([rand_decomp(cont1), rand_decomp(cont2)])
            cont1 -=1
            cont2 +=1
    return res[ np.random.choice( range(len(res)) )]

# Make circuit from a random decomposition with global p_phase_change from Phi = 0º
def circuit(decomp):
    # Extra phase shifter?
    global p_PHS_change, p_extra_BS_PHS, dec_n
    extras = '' #'None'
    if np.random.rand() < p_extra_PHS:
        #Phi = Phi_BS()
        extras = 'PHS_' + str( np.round(Phi_BS(), dec_n )) + ' '
   
    # Terminal nodes
    if decomp == 1:
        res =  [extras + 'Sup._' + str( np.round(Theta_sup(), dec_n ))]
        return res
    
    if decomp == 2 :
        res = [extras + 'Part._' + str(np.round(F(), dec_n ))]
        return res

    # Recursion
    if len(decomp) == 2:
        Theta = Theta_BS()
        Phi = 0.
        if np.random.rand() < p_BS_PHS_change:
            Phi = Phi_BS()
        params = str( np.round(Theta, 2 )) + '_' + str( np.round(Phi, dec_n ))
        res =  [extras + 'BS_' + params , [ circuit(decomp[0]), circuit(decomp[1])]  ]
        return res


########################################################
# Circuit information
#
########################################################
# Number of single and two entangled resources in a decomp.
def n_resources(decomp, res1 = 0, res2 = 0):
    if type(decomp) != list:
        if decomp == 1:
            res1 += 1
        if decomp == 2:
            res2 += 1
    else:
        a1, a2 = n_resources(decomp[0], res1)
        b1, b2 = n_resources(decomp[1], res2)
        res1 += a1 + b1
        res2 += a2 + b2

    return [res1, res2]

########################################################
# Tree information
#
########################################################
# Return the type (T/NT) of a node
def node_type(node):
    if type(node) == list:
        node = node[0]
    res = node.split()
    # Use last element [-1] of split
    if res[-1][:5] == 'Part.' or res[-1][:4] == 'Sup.':
        return 'T'
    else:
        return 'NT'

# Return number of levels of a tree
def depth(tree):
    if node_type(tree) == 'T': #len(tree)== 1: #type(tree) == float 
        res = 0
    else:
        if len(tree[1]) == 1:
            res = 1 + depth( tree[1] )
        elif len(tree[1]) == 2:
            res = 1 + max( depth( tree[1][0]), depth(tree[1][1] ) )
    return res

# Create string from a tree
def expr(tree, dec = -1):
    if node_type(tree) == 'T': #float or len(tree)== 1:
        #if type(tree) == float and dec !=-1:
        #    res = str( round(tree, dec) )
        #else:
        res = str( tree[0] )
    else:
        if len(tree[1]) == 1:
            res = str( tree)
        elif len(tree[1]) == 2:
            res = '<' + expr(tree[1][0], dec) +'|'+  str(tree[0]) +'|' + expr(tree[1][1], dec)+'>'
    return res

# Number of nodes in a tree:
def n_nodes(tree):
    if node_type(tree) == 'T': #type(tree) == float or len(tree)== 1:
        return 1
    if len(tree[1]) == 1:
        return n_nodes(tree[1]) + 1
    else:
        return ( n_nodes(tree[1][0]) + n_nodes(tree[1][1]) + 1 )

# Number of NT nodes in a tree:
def NT_nodes(tree):
    if node_type(tree) == 'T' or len(tree)== 1:
        return 0
    if len(tree)== 1 and node_type(tree[0]) == 'T':
        return 1
    
    if len(tree[1]) == 1:
        return NT_nodes(tree[1]) + 1
    else:
        return ( NT_nodes(tree[1][0]) + NT_nodes(tree[1][1]) + 1 )

# Obtain a sub-tree indexed by a list of indices
def tree_by_idx(tree, l):
    res = tree
    for each in l:
        res = res[each]
    return res

# List of a tree nodes
def tree_idx(tree):
    global my_idx
    my_idx = []
    get_indices(tree)
    return my_idx

# Not to be called. Internal function
def get_indices(tree, res = []):
    global my_idx
    my_idx.append(res)
    #print(tree, res)
    if node_type(tree) == 'T': #type(tree) == float or len(tree)== 1:
        return  res + [0]
    else:
        if len(tree[1]) == 1:
            return get_indices(tree[1], res + [0])
        
        elif len(tree[1]) == 2:
            return [ get_indices(tree[1][0], res + [1,0]) +\
                     get_indices(tree[1][1], res + [1,1]) ]

# How many parts are in a tree (or sub-tree)
def n_sources(tree):
    if node_type(tree) == 'T': #type(tree) == float or len(tree)== 1:
        if type(tree) == list:
            tree = tree[0]
        first = tree.split()[-1]
        if first[:5] == 'Part.':
            return 2
        if first[:4] == 'Sup.':
            return 1
    else:
        return n_sources(tree[1][0]) + n_sources(tree[1][1])

# Number of Phase Shifters in a tree
def n_PHS(tree, res = 0):
    #print(tree)
    if type(tree) != list:
        #print('TRUE')
        if tree.split()[0][:3] == 'PHS':
            res += 1
    else:
        a1 = n_PHS(tree[0],res)
        #print('kk', tree)
        if len(tree) == 2:
            b1 = n_PHS(tree[1],res)
        else:
            b1 = 0
        res += a1 + b1

    return res

# List of variable types and values => var_list[type, values]
def tree_to_vars(tree):
    var_list = []
    idx = tree_idx(tree)
    for each in idx:
        head = tree_by_idx(tree, each)[0]
        # With a previous PHS, head has 2 elements
        head = head.split(' ')
        for h in head:
            h = h.split('_')
            type_ = h[0]
            vals = h[1:]
            vals_float = [float(v) for v in vals]
            vals = vals_float
            type_ = [type_]
            type_.extend(vals)
            var_list.append( type_)
    return var_list

# Modify a tree according to a var_list[type, values]
def vars_to_tree(tree, var_list):
    global dec_n

    idx = tree_idx(tree)
    cont_idx = 0
    cont_vars = 0
    for cont_idx in range(len(idx)):
        head = tree_by_idx(tree, idx[cont_idx])[0]
        elem = var_list[cont_vars]
        # If a previous PHS, elem has 2 elements
        if elem[0] == 'PHS':
            cont_vars += 1
            elem.extend( var_list[cont_vars])
            
        # elem tokenization
        elem_str = ''
        cont_elem = 0
        while (cont_elem < len(elem)):
            e = elem[cont_elem]
            if e == 'PHS':
                e2 = elem[cont_elem+1]
                elem_str += e + '_' + str(round(e2, dec_n)) + ' '
                cont_elem += 1
                e = elem[cont_elem]
            else:
                if type(e) == str:
                    elem_str += e
                else:
                    elem_str += '_' + str(round(e,dec_n))    
            cont_elem += 1

        # Substitute [head + tail] by elem_str in tree
        #print(elem_str)
        head = elem_str
        if len(tree_by_idx(tree, idx[cont_idx])) > 1:
            tail = tree_by_idx(tree, idx[cont_idx])[1]
            sub_tree = [head, tail]
        else:
            sub_tree = [head]

        substitute(tree, idx[cont_idx], sub_tree)
        #print(tree)
        cont_vars += 1
    return tree

# Extract effectively manipulable vars. from vars. list
def extract_eff_vars(vars_):
    res = []
    for each in vars_:
        if each[0] == 'PHS':
            res.append(each[1])
        if each[0] == 'BS':
            res.append(each[1])
            res.append(each[2])    
    return res

# Insert effectively manipulable vars. into vars. list
# Modify original vars. list and returns it.
def insert_eff_vars(eff_vars, vars_):
    cont_eff = 0
    for cont_v in range(len(vars_)):
        if vars_[cont_v][0] == 'PHS':
            vars_[cont_v][1] = eff_vars[cont_eff]
            cont_eff += 1
        if vars_[cont_v][0] == 'BS':
            vars_[cont_v][1] = eff_vars[cont_eff]
            vars_[cont_v][2] = eff_vars[cont_eff+1]
            cont_eff += 2
    return vars_

# Delete a PHS at the head of tree_by_idx(id)
def del_PHS(tree, idx):
    return res

# Intert a PHS at the head of tree_by_idx(id) with default/given Phi angle
def insert_PHS(tree, idx, Phi = None):
    return res

########################################################
# Tree manipulation
#
########################################################
# There's no random tree function.
def tree():
    return res

# Evaluate a quantum circuit tree adding random variables: states and noise
def evaluate(tree):
    #print(tree)
    # Terminal nodes are 1 or 2 states with possible post PHS
    if node_type(tree) == 'T':
        ##if not(node_type(tree) == 'T'):
        #print(tree)
        if type(tree)== list: tree = tree[0]
        ###print('===>', tree)
       
        s = tree.split(' ')
        #print(s)
        op_arg = s[-1]
        op, arg = op_arg.split('_')[0:2]
        #print(arg)
        op_arg = float(arg)

        PHS_arg = None
        if len(s) > 1:
            PHS_arg = float(s[0].split('_')[1])
        #print(PHS_arg)
            
        if op == 'Sup.':
            res = superp_state(op_arg * np.pi/180.)
            if PHS_arg != None:
                return PHS(res, PHS_arg)
            else:
                return res
                
        if op == 'Part.':
            res = part_ent_state(op_arg * np.pi/180.)
            if PHS_arg != None:
                return PHS(res, PHS_arg)
            else:
                return res
    else:
        s = tree[0].split(' ')
        BS_arg = s[-1]
        args = BS_arg.split('_')[1:3]
        BS_arg = [float(each) for each in args]

        PHS_arg = None
        if len(s) > 1:
            PHS_arg = float(s[0].split('_')[1])
      
        res = BS( evaluate(tree[1][0]), evaluate(tree[1][1]), BS_arg[0], BS_arg[1])
        if PHS_arg != None:
            return PHS(res, PHS_arg)
        else:
            return res
        
# Substitute a node in position idx by a tree
# Perform substitution in argument and returned value
def substitute(tree, node_idx, new_node):
    # scape single 'x'
    if type(new_node) == str: new_node = "'" + new_node + "'"
    res = ''
    for each in node_idx:
        res += '[' +str(each)  +']'
    res = 'tree[:]' + res + '= ' + str(new_node)
    #print ('exec:', res)
    exec( res )
    return tree

# TESTS
"""
d = rand_decomp(6)
c = circuit(d)
c_idx = tree_idx(c)
depth(c)
n_nodes(c)
NT_nodes(c)
n_sources(c)
n_PHS(c)
e = evaluate(c)
probs(e)

f = circuit(rand_decomp(3))
f_idx = tree_idx(f)
c = substitute(c, c_idx[2], tree_by_idx(f, f_idx[1]))
"""
########################################################
# Fitness to target and probs.
#
########################################################
# Fitness function
def f_function(target, actual):
    global Lambda
    #sse = np.sum((target - actual)**2)
    ent = Ent_2x2(actual, pos_1, pos_2, pos_neg_1, pos_neg_2)
    probs = actual[pos_1][0] + actual[pos_1][1] + actual[pos_2][0] + actual[pos_2][1]
    res = (ent[0] + ent[1]) + (probs) * Lambda
    objs = [ent, probs]
    return [res, objs]
