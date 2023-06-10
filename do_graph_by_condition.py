"""
Plot of data 
Convergence graph of average of several conditions.

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
#data_labels = ["E", "P(E)"] #['GP (soft op.)', 'Random search']
#['GP (Param. mutation)', 'GP (Tree mutation)'] #'Random search'
data_labels = ["PG -> CMA-ES"] #["E", "P(E)"]

output_folder = './results_cma/'
n_parts = 7

runs = 5
iterations = 100
Mu = 100 #2000
L = 0 #1e-1 # Path -> below
#var = 'Fitness'
#var = 'Entanglement'
var = 'CMA-ES, E' #'Soft operator vs random'

#evolution
title = var + '. N= '+ str(n_parts) #+', Mu= ' + str(Mu) #+', Lambda= ' + str(L)
out_f_name = var.lower() + '_' + str(n_parts) + '_' +'_average__'

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
        #print(len(data))
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
    id3 = -1
    id4 = -2


datas = np.zeros((iterations, 2))
data_folder = './output_cma/n_' + str(n_parts)
#data_folder = './output_nsga/n_6_pc_0.2_Mu_2000'
condition = 'MU_2000_GP_0.5_200_SOFT_2_'

best_m1 = make_big(data_folder, condition)[:, :, id1].mean(axis=0)[:iterations] 
std_m1 = make_big(data_folder, condition)[:, :, id1].std(axis=0)[:iterations]
datas[:,0] = np.array(make_big(data_folder, condition)[:, :, id1].mean(axis=0))[:iterations].T



#data_folder = './output_pg+cma/n_' + str(n_parts)
data_folder = './output_nsga/n_6_pc_0.2_Mu_1000'
condition = 'MU_1000_GP_0.2_100_SOFT_2_'

best_m2 = make_big(data_folder, condition)[:, :, id3].mean(axis=0)[:iterations] 
std_m2 = make_big(data_folder, condition)[:, :, id3].std(axis=0)[:iterations]
datas[:,1] = np.array(make_big(data_folder, condition)[:, :, id1].mean(axis=0))[:iterations].T


#data_folder = './output_mu/n_' + str(n_parts)
data_folder = './output_nsga/n_6_pc_0.2_Mu_2000'
condition = 'MU_2000_GP_0.2_200_SOFT_2_'

best_m3 = make_big(data_folder, condition)[:, :, id3].mean(axis=0)[:iterations] 
std_m3 = make_big(data_folder, condition)[:, :, id3].std(axis=0)[:iterations]
#datas[:,2] = np.array(make_big(data_folder, condition)[:, :, id1].mean(axis=0))[:iterations].T


print(np.mean(best_m1[1:]), np.std(best_m1[1:])) #1:
print(np.mean(best_m2[1:]), np.std(best_m2[1:]))
print(np.mean(best_m3[1:]), np.std(best_m3[1:]))


##np.savetxt(output_folder + '/' + out_f_name  + '.txt', datas, fmt = '%.4f', delimiter = '\t')

print('\nFinal\n', best_m1[-1], std_m1[-1])
print(best_m2[-1], std_m2[-1])
print(best_m3[-1], std_m3[-1])



"""
data_folder = './output_tournament/n_' + str(n_parts)
condition = 'GP_0.5_100_SOFT_2_'
best_m3 = make_big(data_folder, condition)[:, :, id1].mean(axis=0)[:iterations] 
std_m3 = make_big(data_folder, condition)[:, :, id1].std(axis=0)[:iterations]

print(np.mean(best_m1), np.std(best_m1))
print(np.mean(best_m2), np.std(best_m2))

data_folder = './output_mu/n_' + str(n_parts)
condition = 'MU_2000_GP_0.5_200_SOFT_2_'
best_m3 = make_big(data_folder, condition)[:, :, id1].mean(axis=0)[:iterations] 
std_m3 = make_big(data_folder, condition)[:, :, id1].std(axis=0)[:iterations]


data_folder = './output_operator/n_' + str(n_parts)
condition = 'GP_0.5_50_SUBTREE_'
best_m2 = make_big(data_folder, condition)[:, :, id3].mean(axis=0)[:iterations]
std_m2 = make_big(data_folder, condition)[:, :, id3].std(axis=0)[:iterations]



data_folder = './output_mu/n_' + str(n_parts)
condition = 'MU_2000_GP_0.5_200_SOFT_2_'
best_m3 = make_big(data_folder, condition)[:, :, id1].mean(axis=0)[:iterations] 
std_m3 = make_big(data_folder, condition)[:, :, id1].std(axis=0)[:iterations]

print(np.mean(best_m1[1:]), np.std(best_m1[1:]))
print(np.mean(best_m2[1:]), np.std(best_m2[1:]))
print(np.mean(best_m3), np.std(best_m3))
"""

########################################################
# Show
########################################################
plot([best_m1], #, best_m2], #, best_m3], #, best_m3],
     range(0, iterations),
     labels = data_labels,
     err = [std_m1], #, std_m2], #, std_m3], #, std_m3],
     y_sup = 1.0, #1.1
     y_inf = 0.0,
     d_x = 10, #25,
     x_label = 'Generation',
     y_label = 'Fitness', #var,
     title = title,
     loc = 'lower right',
     f_name = output_folder + '/' + out_f_name + '.png')

     





