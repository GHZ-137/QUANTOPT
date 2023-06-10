"""
Quantum optics circuit.
"""
from functions_basis import *

#####
# for each in states_2: print(each.shape)

##############################################################################################
# Some Variables
n_parts_max = 4

# Resources # 38 * 2, 40 * 10
Thetas = 38+ ((np.random.rand(400)-0.5) * 2)

Thetas = list(Thetas)


# Tools
tools = dict()
tools['BS'] = [BS, HWP]
tools['part_ent_state'] = [BS, HWP]
tools['superp_state']   = [BS, HWP, Brewster]

##############################################################################################
# Main flow
np.random.seed(23)
np.random.shuffle(Thetas)

# Source states are two kinds: superp. states / part. ent. states
source_sup = [superp_state( np.pi * Theta / 180.) for Theta in Thetas[:100]]
source_pes = [part_ent_state( np.pi * Theta / 180.) for Theta in Thetas[100:200]]
Thetas = [np.pi * each / 180. for each in Thetas[200:]]


# Conditional Brewster of superp. s
for cont in range(len(source_sup)):
    opt = np.random.rand()
    if opt > .75:
        T = Thetas.pop() #np.pi * 30./ 180. #
        print(T)
        source_sup[cont] = Brewster(source_sup[cont], T)
        

# Conditional HWP of superp. s and p.e.s.
for cont in range(len(source_sup)):
    opt = np.random.rand()
    if opt > .75:
        T = Thetas.pop()
        source_sup[cont] = HWP(source_sup[cont], T)
for cont in range(len(source_pes)):
    opt = np.random.rand()
    if opt > .75:
        T = Thetas.pop()
        source_pes[cont] = HWP(source_pes[cont], T)


# Beam-splitting!
# Lists of available n-part states
states_2 = []
states_3 = []
states_4 = []
n_parts = 0


# Repeat some: 1-1+1. 2- 2+1
# states_2 from (sup.s., sup.s.)
for cont in range(20):
    state_A = source_sup.pop()
    state_B = source_sup.pop()
    T = Thetas.pop()
    state_C = BS(state_A, state_B, T)
    states_2.append( state_C)
        
for cont in range(5):
    state_A = states_2.pop()
    state_B = source_sup.pop()
    T = Thetas.pop()
    state_C = BS(state_A, state_B, T)
    states_3.append( state_C)
    
# Repeat some: 2+2.
# states_4 from (p.e.s, p.e.s.)
for cont in range(5):
    state_A = source_pes.pop()
    state_B = source_pes.pop()
    T = Thetas.pop()
    state_C = BS(state_A, state_B, T)
    states_4.append( state_C)

# Final options: 1-3+1. 2.2+2
# states_4 from (BS_2, BS_2)
for cont in range(5):
    state_A = states_3.pop()
    state_B = source_sup.pop()
    T = Thetas.pop()
    state_C = BS(state_A, state_B, T)
    states_4.append( state_C)

for cont in range(5):
    state_A = states_2.pop()
    state_B = states_2.pop()
    T = Thetas.pop()
    state_C = BS(state_A, state_B, T)
    states_4.append( state_C)

# Calculate fitness of final 4-states
print ('4-states created:', len(states_4))
print(enum_states(4))
for each in states_4:
    print('P:' + '%.2f '* len(probs(each)) % tuple(probs(each)) )
    #print('sum(p(...):' + '%.2f ' % np.sum(probs(each)[0]+probs(each)[3]+probs(each)[-4]+probs(each)[-1]) )
    #print('sum(p(...a):' + '%.2f ' % np.sum(probs(each)[0]+probs(each)[-4]) )
    #print('sum(p(...b):' + '%.2f ' % np.sum(probs(each)[3]+probs(each)[-1]) )
print()
for each in states_4:
    print('P:' + '%.2f ' % np.sum(np.abs(probs(each))) )









    
