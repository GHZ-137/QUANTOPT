# CMA-ES to optimize a circuit parameters:
# Theta (BS), Phi(BS, PHS)
#
# Indicate n_parts in config_experiment
# Paste the circuit into variable 'c'
#
# Requires: pip3 install CMA

from config_experiment import *
from config_quantum import *
from config_GP import *
from functions_states import *
from functions_quantum_trees import *
from functions_plot import *
from cmaes import CMA

## <--
## Update n_parts in config_experiment & L in config_GP


#L = "1e-1"
#n = 6
#L = "1"
run = 1

condition = 'GP'
output_folder = './output_cma' + "/n_" + str(n_parts) + "/" + condition + "_" + str(run)

c=\
['BS_12.65_90.0', [['BS_45.1_0.0', [['BS_72.57_90.0', [['BS_44.91_6.41', [['Sup._42.2'], ['Sup._42.2']]], ['Sup._42.2']]], ['BS_45.03_13.34', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]]

print(f_function(target, probs(evaluate(c))) )

# Variables
n_gen = 100
init_sigma = 1.0 # Sigmas and repetition .5 [.01, .1, 1.
bounds = [0, 90] #[1, 89.]

# Secure original circuit
c_orig = c[:]
c = c_orig[:]
orig_vars = tree_to_vars(c)
for each in orig_vars: print(each)

# Initial values
vars_ = extract_eff_vars(orig_vars)
n_vars = len(vars_)
bounds = np.array( [bounds]* n_vars )
init_vals = np.array( vars_ ) # Faster convergence with vars_as init_vals
#init_vals = np.ones(n_vars) * 0.


# CMAES
my_cma = CMA(init_vals, init_sigma)
my_cma.set_bounds(bounds)

for gen in range(n_gen):
    evs = []
    for each in range(my_cma.population_size):        
        c = c_orig[:]
        all_vars = tree_to_vars(c) # EXTRACT EXPLICITELY

        s = my_cma.ask()
        
        insert_eff_vars(s, all_vars)
        c = vars_to_tree(c, all_vars)
        ev = f_function( target, probs(evaluate(c)) )[0]
        #ev = f_function( target, probs(evaluate(c)) )[1][0]
        #ev = np.sum(ev)
        
        evs.append( (s, -ev) )


    my_cma.tell(evs)    
    print('%d, %.2f' % (gen, np.max(ev)))
    
# Print
for each in tree_to_vars(c): print(each)

print(f_function(target, probs(evaluate(c))) )

# Save image
diag = probs( evaluate( c ) )
ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
title = condition + '. Fit:' + ('%.4f' % ev) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])
size = int(np.sqrt(len(diag)))
# If Odd n_parts
if (n_parts % 2 != 0):
    diag2 = np.zeros((size+1)**2)
    diag2[: len(diag)] = diag
    diag = diag2
    size = np.int(np.sqrt(len(diag)))
diag = diag.reshape((size,size))
plt_bar3d(output_folder + ".png", diag.T, title, z_lim = 0.3)

output_folder = './output_cma' + "/n_" + str(n_parts) + "/Best_" + condition + "_" + str(run) + ".txt"
f = open(output_folder, 'w')
f.write(str(c))
f.close()

        


