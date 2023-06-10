"""
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
from functions_NSGA import *

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

in_folder = "./output_nsga/n_6_pc_0.2_Mu_2000"
out_folder = "./results_Pareto_6_opt"

n_parts = 6
Mu = 2000
runs = 5

iterations = 100

var = 'S'
title = var + '. N= '+ str(n_parts) + ', Mu= ' + str(Mu)
out_f_name =  var.lower() + '_' + str(n_parts) + '_JOINED'

########################################################
# DATA. Read the files.
#
########################################################
f = open(in_folder + '/1_pareto_' + str(iterations),"br")
data = np.array( pickle.load(f) )
print(data.shape)

for cont in range(2, runs+1):
    f = open(in_folder + '/' + str(cont) + '_pareto_' + str(iterations),"br")
    d = np.array( pickle.load(f) )
    print(d.shape)
    data = np.vstack((data,d))
    print(data.shape)

join = fronts(data)[0]
join = np.array(join)    
########################################################
# Calculate S
########################################################
data = [join]
S_ = []
plt.clf()
for cont in [0]:
    p = np.array(data[cont])
    # Sort by increasing Y
    p = p[np.argsort(p[:,0])]
    # Reverse by Y(E)
    p = p[::-1]
    S_.append( S(p) )
    
    pol_p = [[0,0] , [0, p[0,0] ]]   
    for cont2 in range(len(p)-1):
        y, x = p[cont2]
        next_y = p[cont2+1][0]
        pol_p.append( [x,y] )
        pol_p.append( [x,next_y] )
        
    #pol_p.append( [x,0] )
    pol_p = np.array(pol_p)
    #print(cont, len(data[cont]), S(p))
    k = 0.1 + (float(cont) / (len(data)+2))
    plt.fill(pol_p[:,0], pol_p[:,1], color = 'blue', alpha = 1.0 )

plt.scatter(data[-1][:,1], data[-1][:,0], s=85, color = 'red')
plt.ylim(0,2)
plt.xlim(0,1); #2
plt.xlabel("P(E)")
plt.ylabel("E")
plt.title('Pareto evolution. N=' + str(n_parts) + '. '+ str(iterations) + ' generations')
 
plt.savefig(out_folder + '/' + out_f_name + '_joined')


dd
########################################################
# Show
########################################################
plot([S_],
     range(0, iterations),
     labels = data_labels,
     err = [],
     y_sup = 1.0,
     y_inf = 0.0,
     d_x = 10, #25,
     x_label = 'Generation',
     y_label =  var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '.png',
     loc = "lower right")


