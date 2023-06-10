"""
NSGA functions
"""

############################################
# Imports
############################################
import numpy as np
from functions_plot import *

# Divide a list of pairs of objectives in non-dominated fronts
# Parameter: a list of pairs of objectives
# Return: a list of lists of pairs of objectives
# sorted right to left
def fronts(objs_l):
    objs = np.array(objs_l)
    fronts = []

    # Staircase algorithm
    while (len(objs) > 0):
        # Sort by 2nd obj
        vals = np.sort(objs[:,1])[::-1]
        vals = np.unique(vals)[::-1]
        #print('\n', objs, vals)
       
        # Keep highest in 1st obj, going right to left by 2nd obj
        f = []
        max_ = 0
        for v in vals:
            ###print(v)
            
            idx = np.where(objs[:,1]== v)[0]
            selec = objs[idx]
            res_idx = np.argsort(selec[:,0])[::-1][0]
            obj_idx = idx[res_idx]
            if objs[obj_idx,0] >= max_:
                max_ = objs[obj_idx,0]
                ###print("  M", max_)
                
                f.append( objs[obj_idx].tolist() )
                objs = np.delete(objs, obj_idx, axis = 0)

        fronts.append(f)
    return fronts


# Return crowding distance of each element in the front.
# Parameter: a list of pairs of objectives
# Return: a list of distances
def crowd_dist(objs_l):
    # Distance is the enclosing box area between adjacent points
    # in the (sorted) front
    res = []
    objs = np.array(objs_l)

    for cont in range(len(objs)):
        if cont == 0 or cont == (len(objs)-1):
            a = float("Inf")
        else:
            idx1 = cont - 1
            idx2 = cont +1
            
            a = abs(objs[idx1][0] - objs[idx2][0]) *\
                 abs(objs[idx1][1] - objs[idx2][1])
             
        res.append(a)
    return res

# Create a list of indices of individuals take from a list of objectives
# Parameter: front, a list of lists
# Parameter: a list of objectives pairs
# Return: list of front indices of objs_l
def front_idx(f, objs_l):
    res = []
    objs = np.array(objs_l)
    
    for each in f:
        ##print(each, np.where(objs == each)[0][1]  )
        i = np.where(objs == each)[0][1
                                      ]
        res.append(i)
    return res

def flatten(list_):
    res =[item for sub_list in list_ for item in sub_list]
    return res

# Create arrays of fronts and distances of a list of bi-objectives
# and lists of front indices, distances and fronts
# Argumetns: objs and max size of Pareto front to prevent overgrow
def front_crowd_arr(objs_l, F0_max_size = float('Inf')):
    f = fronts(objs_l)
    ##print('front 1:', len(f[0]))
    
    fronts_idx = [front_idx(each, objs_l) for each in f]
    fronts_crowd = [crowd_dist(each) for each in f]

    F0_size = len(fronts_idx[0])

    #print(len(f[0]))
    
    # Truncate first front if size exceeded
    if (F0_size > F0_max_size):     
        #sel_id = [0]
        #sel_id += list(np.random.choice(range(1, F0_size-1), F0_max_size -2))
        #sel_id += [-1]

        # Truncate by decreasing crowding
        dist_idx = np.argsort(fronts_crowd[0] )[::-1]
        sel_id = dist_idx[:F0_max_size]

        #print(sel_id)
        
        fronts_idx[0] = [fronts_idx[0][each] for each in sel_id]
        fronts_crowd[0] = [fronts_crowd[0][each] for each in sel_id]
        # Truncate fronts of objectives
        f[0] = [f[0][each] for each in sel_id]
        
    #print(len(f[0]))

    front_arr = np.zeros(len(objs_l))
    crowd_arr = np.zeros(len(objs_l))
    for c1 in range(len(fronts_idx)):
        for c2 in range(len(fronts_idx[c1])):
            front_arr[ fronts_idx[c1][c2] ] = c1
            crowd_arr[ fronts_idx[c1][c2] ] = fronts_crowd[c1][c2]


    
    return [front_arr, crowd_arr, fronts_idx, fronts_crowd, f]

"""
# Tests
np.random.seed(26)
objs_l =  np.random.rand(8,2) * 10
f = fronts(objs_l)

my_fronts_idx = [front_idx(each, objs_l) for each in f]
my_fronts_crowd = [crowd_dist(each) for each in f]

print(my_fronts_idx)
print(my_fronts_crowd)

front_arr, crowd_arr = front_crowd_arr(objs_l)

print (front_arr)
print (crowd_arr)

ddd

front_arr = np.zeros(len(objs_l))
crowd_arr = np.zeros(len(objs_l))
for c1 in range(len(my_fronts_idx)):
    for c2 in range(len(my_fronts_idx[c1])):
        front_arr[ my_fronts_idx[c1][c2] ] = c1+1
        crowd_arr[ my_fronts_idx[c1][c2] ] = my_fronts_crowd[c1][c2]


dd

for each in f:
    d = crowd_dist(each)
    print(each)
    print(d, "\n")

plt.clf()
for each in f:
    each = np.array(each)
    plt.scatter(each[:,1], each[:,0])
plt.savefig('test/Pareto.png')

front_idx(f[3], objs)
"""
