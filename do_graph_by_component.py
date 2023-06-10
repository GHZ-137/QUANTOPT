"""
Plot of data 
Convergence graph of a single condition.

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
data_labels = ['n(1)', 'n(2)', 'PHS'] 
output_folder = './results/'
n_parts = 6

runs = 5
iterations = 100
Mu = 2000
L = 0 # Path -> below
var = 'Components'

title = var + ' evolution. n ='+ str(n_parts)+', Mu= ' + str(Mu) +', Lambda= ' + str(L)
out_f_name = var.lower() + '_' + str(n_parts) + '_' + str(L)


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
data_folder = './output_MO_0/n_' + str(n_parts)
condition = 'GP_'
best_m1 = make_big(data_folder, condition)[:, :, 4].mean(axis=0)[:iterations]
best_m2 = make_big(data_folder, condition)[:, :, 5].mean(axis=0)[:iterations]
best_m3 = make_big(data_folder, condition)[:, :, 6].mean(axis=0)[:iterations]

########################################################
# Show
########################################################
plot([best_m1, best_m2, best_m3],
     range(0, iterations),
     labels = data_labels,
     #err = [best_std, avg_std, worst_std],
     y_sup = 10., #1
     y_inf = 0.0,
     d_x = 10, #25,
     x_label = 'Generation',
     y_label = 'n',
     title = title,
     f_name = output_folder + '/' + out_f_name + '.png')

     





