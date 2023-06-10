"""
Create a random quantum optics circuit for n parts.

Sources: superp. states(Theta), with maximum number:
   n_max_parts // len(basis)
   
Tools:
  BS, PBS, POL_V, POL_H, POL_45, DETECT, HERALD

"""
from functions_basis import *
from functions_states import *
from functions_quantum_trees import *

from config_quantum import *
from config_experiment import *

##################################################################
# Main program
##################################################################
# Variables in config_quantum
np.random.seed(31)
#a = rand_decomp(n_max_sources)
a = [[[2,1],2],1]
b = circuit(a)
c = evaluate(b)

# Test detectors
"""
print(n_detectors(b))
rnd_detect = rnd_detectors(b, 3, 3)
bb = set_detectors(b, rnd_detect)
print(n_detectors(bb))
c = adjust_detectors(bb, 4, 3)
cc = set_detectors(bb, c)
print(n_detectors(cc))
"""

# Evaluate
# states and targets in config_quantum

print('Decomp. of '+ str(n_parts) + ':\n', a)
print('\nCircuit:\n', a )

print('Nodes:',n_nodes(b))
print('No term. nodes:',NT_nodes(b))
print('Depth:',depth(b), '\n')

diag = probs(c)
fit = f_function(target, diag)
print('Fit: %.2f' % fit[0])
print('Prob: %.2f' % fit[1][1])
