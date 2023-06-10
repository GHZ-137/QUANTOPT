"""
Pareto front evolution in single image.
Fronts are sorted: E, y --  p(E), x decreasing

"""
########################################################
# Inclusions
########################################################
#from config_GP import *
import numpy as np
import pickle
from functions_plot import *

########################################################
# Variables
########################################################
out_folder = "./output_nsga/n_7" 
n_parts = 7
Mu = 1000
run = 5
iterations = 100 #100
step= 10 #10

#out_folder = out_folder + "/n_" +str(n_parts) #+ "_"
in_folder = out_folder

title = 'Pareto front evolution. N ='+ str(n_parts)+', Mu= ' + str(Mu)
out_f_name = '_Pareto_evolution_' + str(run)

########################################################
# DATA. Read the files.
#
########################################################
data = []
legend = []
range_l = list(range(0, iterations+1, step))
range_l[0] = 1
#range_l = [44]

for cont in range_l:
    f = open(in_folder + '/' + str(run) + '_pareto_' + str(cont),"br")
    d = pickle.load(f)
    d = np.array(d)
    data.append(d)
    legend.append(str(cont))



########################################################
# Show
########################################################
plt.clf()
plt.ylim(0,2)
plt.xlim(0,1); #2
plt.xlabel("P(E)")
plt.ylabel("E")
plt.title('Pareto evolution. N=' + str(n_parts) + '. '+ str(iterations) + ' generations')
             
for cont in list(range(len(data)))[::-1] :
    p = np.array(data[cont])
    # Sort by increasing Y
    p = p[np.argsort(p[:,0])]
    # Reverse by Y(E)
    p = p[::-1]
    
    pol_p = [[0,0] , [0, p[0,0] ]]
    for cont2 in range(len(p)-1):
        y, x = p[cont2]
        next_y = p[cont2+1][0]
        pol_p.append( [x,y] )
        pol_p.append( [x,next_y] )
    pol_p.append( [x,0] )
    pol_p = np.array(pol_p)
    
    #print(pol_p)
    a = 0.1 + ((len(data)-float(cont)) / (len(data)+2))
    c1 = (1,0,0, a)
    c2 = (0,1,0, a)
    c3 = (0,0,1, a)
    cc = [c1, c2, c3]
    color = cc[cont % 3]
    color = 'blue'

    plt.fill(pol_p[:,0], pol_p[:,1], color = color), alpha = 0.07 ) #0.1 + (float(cont) / (len(data)+2)) )
    #plt.plot(p[:,1], p[:,0],color = 'b', alpha = 0.5 ) #0.1 + (float(cont) / (len(data)+2)) )

#plt.legend(legend)
plt.savefig(out_folder + '/' + out_f_name)
print(out_folder + '/' + out_f_name)
