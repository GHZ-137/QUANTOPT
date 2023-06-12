"""
Use case comparing single/multiobjective optimization
for information transmission
"""

############################################
# Imports
############################################
import numpy as np, cv2
#from sklearn.datasets import load_sample_image
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt


from functions_quantum_trees import *
from functions_plot import *
from config_experiment import *

############################################
# Variables
############################################
condition = 'PG+CMA_5'
#condition = 'PG+CMA_4'
#condition = 'PG+CMA_3'
#condition = 'PG+CMA_2'
#condition = 'PG+CMA_1'
#condition = 'PARETO'


output_folder = './test/case'
name = '_probs_'
name += condition + '_MU_2000_GP_0.2_200_SOFT_2_3'

c = ['BS_76.78_5.19', [['BS_34.56_1.58', [['BS_55.6_0.46', [['PHS_87.05 Part._0.95'], ['Sup._42.2']]], ['Sup._42.2']]], ['BS_54.11_3.71', [['BS_44.75_4.1', [['Sup._42.2'], ['Sup._42.2']]], ['Sup._42.2']]]]]
#c = ['BS_80.79_88.74', [['BS_77.73_0.27', [['BS_45.01_0.74', [['BS_45.02_0.79', [['Sup._42.2'], ['Sup._42.2']]], ['BS_45.08_1.69', [['Sup._42.2'], ['Sup._42.2']]]]], ['BS_17.85_0.18', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]]
#c = ['PHS_12.53 BS_0.1_20.64', [['BS_44.15_0.92', [['BS_2.31_87.89', [['BS_72.96_84.41', [['BS_45.22_0.31', [['Sup._42.2'], ['Sup._42.2']]], ['Sup._42.2']]], ['BS_45.25_2.72', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]], ['Sup._42.2']]]
#c = ['BS_83.48_7.29', [['BS_0.05_29.25', [['PHS_7.55 BS_40.25_6.61', [['BS_44.47_8.18', [['PHS_84.92 Sup._42.2'], ['Sup._42.2']]], ['Sup._42.2']]], ['PHS_17.18 BS_37.24_0.79', [['BS_46.4_1.71', [['PHS_84.94 Sup._42.2'], ['Sup._42.2']]], ['Sup._42.2']]]]], ['Sup._42.2']]]
#c =['BS_85.29_31.19', [['BS_44.84_4.74', [['BS_89.94_88.87', [['BS_44.69_0.08', [['Sup._42.2'], ['Sup._42.2']]], ['BS_44.63_2.07', [['Sup._42.2'], ['Sup._42.2']]]]], ['BS_44.19_4.49', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]]

#c= ['BS_49.74_3.97', [['BS_4.96_85.57', [['BS_88.64_88.81', [['BS_45.38_1.26', [['Sup._42.2'], ['Sup._42.2']]], ['BS_44.63_0.0', [['Sup._42.2'], ['Sup._42.2']]]]], ['BS_69.4_26.21', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]]



def calc_diag(circuit):
    global target
    diag = probs( evaluate( circuit ) )
    ent = Ent_2x2(diag, pos_1, pos_2, pos_neg_1, pos_neg_2 )
    res = ent[0] + ent[1]
    P_E = diag[pos_1][0] +  diag[pos_1][1] +  diag[pos_2][0] +  diag[pos_2][1]

    print('E:    %.4f / %.4f' % (ent[0], ent[1]) )
    print('P(E): %.4f\t' % P_E)

    return diag, ent, P_E

# Calculate probs. as density matrix diagonal
diag, ent, P_E = calc_diag(c)
ent = ent[0]+ent[1]

# Enumerate states. V-1, H-0
states = enum_states(n_parts)

# Greyscale image
img_shape = (100, 100)
img = rgb2gray( data.cat() )
img = resize(img, img_shape, anti_aliasing=True)
img = (img * 255).astype(int)
cv2.imwrite(output_folder + '/' +'cat.png', img)

# Prepare variables
n_users = n_parts - 2
cont = 0
rec_img = []
for c in range(n_users - 1):
    rec_img.append( np.zeros(img_shape) )


# If coincide of extremes
# and if sender (first position user) checks its bit value == bit to send
# => send bit to rest of users

for row in range(img_shape[0]):
    for col in range(img_shape[1]):
        
        byte_ = bin( img[row][col] )[2:]
        #int(bin(8)[2:], 2)
        
        # For each bit
        rec_bit = []
        for c in range(n_users - 1):
            rec_bit.append("")
        
        for bit_ in byte_:
            
            # Repeat until coincidence of extremes and bit to send
            coincidence = False
            while not(coincidence):
                # Draw a random result
                idx = np.random.choice( range(len(diag)), 1, p = diag)[0]
                res = states[idx]
                cont += 1
                # If coincidence of extremes and bit to sned
                if res[0] == res[-1]:
                    if (res[1] == 'V' and bit_ == '1') or \
                       (res[1] == 'H' and bit_ == '0'):
                        coincidence = True
         
            # Send to users
            for c in range(n_users - 1):
                my_res = res[2+c]
                if my_res == 'V': my_res = '1'
                if my_res == 'H': my_res = '0'
                rec_bit[c] = rec_bit[c] + my_res
                #print( rec_bit[c] )
                
            # Important
            coincidence = False

        # Convert received bits to byte
        for c in range(n_users - 1):
            res_byte = int(rec_bit[c], 2)
            rec_img[c][row,col] = res_byte
            #print( rec_bit[c] )

        print(row)

# Analyze received images
print(condition)
print('E:    %.4f' % ent )
print('P(E): %.4f\t' % P_E)
print('Trials: %d' %cont)
print('RMSE: ')
for c in range(n_users - 1):        
    cv2.imwrite(output_folder + '/'+ condition +'_received' + str(c+1)+ '.png', rec_img[c])            
    rmse = np.sqrt( np.sum((img - rec_img[c])**2) / np.prod(img_shape) )
    print('%.4f' % (rmse /255))
    
    







