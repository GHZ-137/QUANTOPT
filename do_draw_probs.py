"""
Plot 3d probabilities evaluating a circuit
"""

############################################
# Imports
############################################
import numpy as np
import matplotlib.pyplot as plt, cv2
from mpl_toolkits.mplot3d import Axes3D
from skimage.draw import *

from functions_quantum_trees import *
from functions_plot import *
from config_experiment import *

############################################
# Variables
############################################

output_folder = './output_mu/n_6/'
name = '_probs_'
name += 'MU_1000_GP_0.5_200_SOFT_2_3'



c = \
['BS_3.9_85.14', [['BS_46.74_0.0', [['BS_0.1_65.91', [['BS_44.58_0.36', [['Sup._42.2'], ['Sup._42.2']]], ['BS_44.97_1.33', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]], ['Sup._42.2']]]

def arrange_probs(circuit):
    global target
    diag = probs( evaluate( circuit ) )
    ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
    res = ent[0] + ent[1]
    print('E:')
    for each in ent: print('%.4f\t' % each)
    title = 'GP. Fit:' + ('%.4f' % res) + '. E:' + ('%.4f' % ent[0]) + ', '+ ('%.4f' % ent[1])

    size = int(np.sqrt(len(diag)))
    # If Odd n_parts
    if (n_parts % 2 != 0):
        diag2 = np.zeros((size+1)**2)
        diag2[: len(diag)] = diag
        diag = diag2
        target2 = np.zeros((size+1)**2)
        target2[: len(target)] = target
        target = target2
            
        size = np.int(np.sqrt(len(diag)))
        
    diag = diag.reshape((size,size))
    t = target.reshape((size, size))
    return [title, diag, title]

title, diag, title = arrange_probs(c)
plt_bar3d(output_folder + name + ".png", diag.T, title, z_lim = 0.3)


