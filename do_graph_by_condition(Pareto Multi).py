"""
SEVERAL CONDITIONS
Scaling independent measures of MOEA:
S - size of the dominated space
    Area( U{ Rect[(0,0), (f(x1), f(x2))] } )

D(A,B) - coverage difference:
size of space weakly dominated by A but not by B
    S(A+B) - S(B)
"""
########################################################
# Inclusions
########################################################
#from config_GP import *
import numpy as np
import pickle
from functions_plot import *

########################################################
# Metrics
# objs:  P(E),E #{E, P(E)}
# vars: P(E) (x), E (y)
########################################################
def S(front):
    prev_x = 0 #front is [[y,x]...]
    res = 0
    for each in front:
        dx =  (each[1] - prev_x)
        y = each[0]
        res += dx * y
        prev_x = each[1]
    #print( res, front )
    return res

def D(front1, front2):
    return res

########################################################
# Variables
########################################################
data_labels = ["P(m) = 0.8","P(m) = 0.5","P(m) = 0.2"]  #["Mu = 500", "Mu = 1000", "Mu = 2000"]
out_folder = "./results_Pareto_6"

n_parts = 6
Mu = 1000
run = 5

iterations = 100
step = 1

var = 'S'
title = var + '. N= '+ str(n_parts) + ', Mu= ' + str(Mu) #+ ', run= ', str(run)
out_f_name =  var.lower() + '_' + str(n_parts) + '_PC'

range_l = list(range(1, iterations+1, step))
range_l[0] = 1

########################################################
# DATA. Read the files.
#
########################################################
def calc_S(in_folder, run):
    # prepare data
    data = []
    for cont in range_l:
        f = open(in_folder + '/' + str(run) + '_pareto_' + str(cont), "br")
        d = pickle.load(f)
        # close polygon at x=0, left <- right
        d = [ [0, d[0][1] ] ] + d
        d = np.array(d)
        data.append(d)

    # dalculate S
    S_ = []
    for cont in range(len(data)):
        p = np.array(data[cont])
        # Sort by increasing Y
        p = p[np.argsort(p[:,0])]
        # Reverse by Y(E)
        p = p[::-1]
        S_.append( S(p) )
    return S_
    
in_folder = "./output_nsga/n_7" #6_pc_0.2_Mu_500"
S1 = []
for r in range(1, 6): S1.append(calc_S(in_folder, r))

in_folder = "./output_nsga/n_6_pc_0.2_Mu_1000"
S2 = []
for r in range(1, 6): S2.append(calc_S(in_folder, r))

in_folder = "./output_nsga/n_6_pc_0.2_Mu_2000"
S3 = []
for r in range(1, 6): S3.append(calc_S(in_folder, r))

m1 = np.mean(S1, axis=0)
s1 = np.std(S1, axis=0)
m2 = np.mean(S2, axis=0)
s2 = np.std(S2, axis=0)
m3 = np.mean(S3, axis=0)
s3 = np.std(S3, axis=0)

dd
########################################################
# Show
########################################################
plot([m1, m2, m3],
     range(0, iterations),
     labels = data_labels,
     err = [s1, s2, s3],
     y_sup = 0.5,
     y_inf = 0.0,
     d_x = 10, #25,
     x_label = 'Generation',
     y_label =  var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '.png',
     loc = "lower right")


