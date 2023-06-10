"""
Plot of data 
Convergence graph of individual runs of two conditions

Jose Luis Rubén García-Zurdo
"""
########################################################
# Inclusions
########################################################
#from config import *
import numpy as np
from functions_plot import *

########################################################
# Variables
########################################################
data_labels = ['Random search', 'GP (Tree mutation)'] 
output_folder = './results/'
n_parts = 6

runs = 3
iterations = 100
Mu = 1000
L = 0 #1e-1 # Path -> below
#var = 'Fitness'
#var = 'Entanglement'
var = 'operator_tree_rnd'

#title = 'GP with local search ' +var.lower() + ' evolution. n ='+ str(n_parts)+', Lambda= ' + str(L)
title = var + ' evolution. n ='+ str(n_parts)+', Mu= ' + str(Mu) #+', Lambda= ' + str(L)
out_f_name = var.lower() + '_' + str(n_parts) + '_' +'_arr'


# Make big data-array
def make_big(data_folder, condition):
    res = []
    for cont1 in range(runs): #
        f_name = condition + str(cont1+1)
        f = open(data_folder + '/' + f_name + '.txt')
        data = f.readlines()[1:]
        f.close()
    
        # Convert to array
        for cont in range(len(data)):
            data[cont] = data[cont].split()
    
        data = np.array(data).astype(np.float)###[:iterations,:]
        res.append(data)
    res = np.array(res)
    return res

########################################################
# DATA. Read the files.
#
########################################################
if True: #var == 'Fitness':
    id1 = 0 #0
    id2 = 1 #1


data_folder = './output_operator/n_' + str(n_parts)
condition = 'GP_0.5_50_SUBTREE_' #2_'
best_m2 = make_big(data_folder, condition)[:, :, id1][:,:iterations] #.mean(axis=0)[:iterations]

data_folder = './output_operator/n_' + str(n_parts)
condition = 'BA_0.5_50_'
best_m1 = make_big(data_folder, condition)[:, :, id2][:,:iterations] #.mean(axis=0)[:iterations] 


########################################################
# Show
########################################################
plot_multi(\
     [best_m1, best_m2],
     range(0, iterations),
     labels = data_labels,
     y_sup = 1.0,
     y_inf = 0.0,
     d_x = 10, #25,
     x_label = 'Generation',
     y_label = var ,
     title = title,
     f_name = output_folder + '/' + out_f_name + '.png')

     





