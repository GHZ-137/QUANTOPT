# Optimization
save = True
optimization = 'Maximize' #'Minimize
use_multi = False #True #PONER A FALSE
n_procs = 1 #3, 10 1 

# Multi-objective weight for positive probs. in aggregate f.
Lambda = 0. #1  #0 for NSGA #0.1 #

# CMA Local search
# Generations between local search optimization:
# 1 (always), 0 (No), -1 (only when improvement)
freq_cma = 0 #-1 #5 +1
frac_ind_cma = 0.2   # << == 20%

init_sigma = 1.0
bounds_orig = [0, 90] #[0.1, 90] for Theta
max_ite_cma = 3  # << == 3

# NSGA
do_NSGA = True
do_NSGA_CMA = False #True
Pareto_graph_limit = 2 #1.5

# Population
Mu_0 = 1000 #300 # 2000 #1000 for NSGA
Mu = 1000 #100 #500, 1000, 2000  << = 2000
Lambda_size = 1000 # Mu << Only different for NSGA

p_c = 0.2 #5 #0.8, 0.5, 0.2  << == 0.5 PG.  0.2 NSGA
p_m = 0.8 #1 - p_c # 0.1 0.5 0.8
p_r = 0.0 #0. # Excluded

soft_mutation = True #False #True << ==  True
soft_all = False
soft_sigma = 2 #0.5 1 2  << == 2

elitism = 'Yes'

selection = 'tournament'
p_Gamma = 0.1 #2.5% 5% 10% << == 0.1
Gamma = int(round(p_Gamma * Mu)) #Tournament size 2,5,10,50,100,200
max_ite = 100 #100 << == 100 for 4,5;  200 for 6,7


