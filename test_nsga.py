"""
NSGA tests.
"""
from functions_NSGA import *

# Data
np.random.seed(26)
objs_l =  np.random.rand(14,2) * 10
objs_l = objs_l.round(2)
f = fronts(objs_l)

# Fronts
print('Fronts:')
for each in f:
    print(' ', each )
    for each2 in each:
        print('   ', each2)
        
# Distances
print('\nDistances:')
for each in f:
    d = crowd_dist(each)
    print(' ', d )
    for each2 in d:
        print('   ', each2)

# Draw fronts
plt.clf()
for each in f:
    each = np.array(each)
    plt.scatter(each[:,1], each[:,0])
plt.savefig('test/Pareto.png')

# Array of front indices
print('\nIndices of:')
for each in objs_l:
    print(' ', each )
 
print('\n F[0] indices:', front_idx(f[0], objs_l))


fronts_idx = [front_idx(each, objs_l) for each in f]
fronts_crowd = [crowd_dist(each) for each in f]

print(fronts_idx)
print(fronts_crowd)

front_arr, crowd_arr = front_crowd_arr(objs_l)

print ('\n', front_arr)
print (crowd_arr)
