import numpy as np
from copy import deepcopy

from config_quantum import *
#from config_experiment import *

# Combinatorial enumeration of states for a number of parts using global basis
def enum_states(n = 2):
    global basis
    res = deepcopy(basis)
    for cont1 in range(n-1):
        for cont2 in range(len(res)):
            r = []
            for b in basis:
                r.append(res[cont2] + b) 
            res[cont2] = r
        res = list( np.array(res).flatten() )
    return res

states_2 = enum_states(2)
states_4 = enum_states(4)

# Indices of a part number in the list of states
def part_idx_from_states(states, part = 1):
    n = len(states[0])
    res = []
    for cont in range(len(states)):
        res.append( cont * n + part -1) 
    return res
