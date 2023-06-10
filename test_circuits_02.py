"""
Test a given circuit

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

c = ['PHS_34.53 BS_47.37_47.34', [['PHS_49.33 Part._0.95'], ['PHS_40.81 Part._0.95']]]

c = ['PHS_84.53 BS_47.37_47.34', [['PHS_49.33 Part._0.95'], ['PHS_40.81 Part._0.95']]]


# Evaluate
# states and targets in config_quantum
print('Nodes:',n_nodes(c))
print('NO term. nodes:',NT_nodes(c))
print('Depth:',depth(c), '\n')


diag = probs(evaluate(c))
fit = f_function(target, diag)
print('Fit: %.4f' % fit[0])
print('Prob: %.4f' % fit[1][1])
print(fit)
