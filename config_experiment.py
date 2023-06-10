# Experiment variables
#
import numpy as np
from functions_states import *

# N of particles in target state
n_parts = 7 #8 #7 10 12 14
h_heralds = 2

# Decimal number in tools parameters
dec_n = 2

# Repetitions and iterations during random search
#repetitions = 1 ##10 #1 #10 # Not used
#max_ite = 2000 #* 50 # 100 #2000

# Function to create lists of target positions for number of particles and a list of herald positions
# 4 combinations:
#  - Similar heralds = 'V...V' , 'H...H'
#  - Similar/dffierent measured = '.VVV.' , '.HHH.'

def target_pos(n_parts, herald_pos):
    states = enum_states(n_parts)

    # States 1,2 
    e1 = ['V']* n_parts
    for h in herald_pos:
        e1[h] = 'H'
    e2 = ['V']* n_parts
    for h in herald_pos:
        e2[h] = 'V'
    # States 3,4
    e3 = ['H']* n_parts
    for h in herald_pos:
        e3[h] = 'H'
    e4 = ['H']* n_parts
    for h in herald_pos:
        e4[h] = 'V' 

    l_e1 = ''.join(str(e) for e in e1)
    l_e2 = ''.join(str(e) for e in e2)
    l_e3 = ''.join(str(e) for e in e3)
    l_e4 = ''.join(str(e) for e in e4)
    pos1 = [states.index(e) for e in [l_e2,l_e4]]
    pos2 = [states.index(e) for e in [l_e1,l_e3]]
    ##print(l_e2, l_e4, l_e1, l_e3)
    pos_neg1 = list( range(pos1[0]+2, pos1[1], 2) )
    pos_neg2 = list( range(pos2[0]+2, pos2[1], 2) )
    return [pos1, pos2, pos_neg1, pos_neg2]

# Function to create the target state
def make_target(n_parts, herald_pos):
    target = np.zeros(2**n_parts)
    pos1, pos2, pos_neg1, pos_neg2 = target_pos(n_parts, herald_pos)
    pos = pos1 + pos2
    for each in pos:
        target[pos] = 1.
    target /= np.sum(target)
    return [target, pos, pos1, pos2, pos_neg1, pos_neg2]

# Create the target and list of positions
target, pos, pos_1, pos_2, pos_neg_1, pos_neg_2 =\
        make_target(n_parts, [0, n_parts-1])

