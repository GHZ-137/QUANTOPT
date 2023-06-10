"""
Plot functions
"""

############################################
# Imports
############################################
import numpy as np
import matplotlib.pyplot as plt, cv2
from mpl_toolkits.mplot3d import Axes3D
from skimage.draw import *
from functions_quantum_trees import *

BS_img = cv2.imread('./_data/BS.png', 0)
PHS_img = cv2.imread('./_data/PHS.png', 0)
PHS_2_img = cv2.imread('./_data/PHS_2.png', 0)
Superp_img = cv2.imread('./_data/Superp.png', 0)
Part_img = cv2.imread('./_data/Part.png', 0)
size_i = 30
h_size_i = size_i//2

# Convert parameters string to readable
def chain(l):
    l = l.split('_')
    if l[0] == "Sup." or l[0] == "Part.":
        return ''
    res = '(' + str(float(l[1])) #l[0] + 
    if len(l) == 3:
        res += ', '+ str(float(l[2]))+')'
    else:
        res += ')'
    return res

# Draw tools
def tools_img(tree, row, col, img):
    global cont_r, cont_c, font, font_scale
    if node_type(tree) == 'T':
        new_row = int(row + cont_r*inc_row)
        new_col = int(col - cont_c *inc_col)
    else:
        if len(tree[1]) == 1:
            BS = tree[1]
            new_row = int(row + cont_r*inc_row)
            new_col = int(col - cont_c *inc_col)
         
        elif len(tree[1]) == 2:
  
            params = tree[0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
            else:
                BS = params[0]
    
            if BS[:2] == "BS":
                img[row - h_size_i+0: row + h_size_i+0, col - h_size_i:col + h_size_i] =\
                            BS_img[:size_i,:size_i]
                            
            new_row = int(row+ cont_r*inc_row)
            new_col = int(col- cont_c *inc_col)
            
            params = tree[1][0][0]
            params = params.split(" ")
            
            if len(params) > 1:
                PHS, BS = params
               
                #<=
                img[new_row - h_size_i-40: new_row + h_size_i-40, new_col - h_size_i+36:new_col + h_size_i+36] =\
                            PHS_img[:size_i,:size_i]
    
            else:
                BS = params[0]

            
            if BS[:2] == "Su":
                img[new_row - h_size_i+0: new_row + h_size_i+0, new_col - h_size_i:new_col + h_size_i] =\
                            Superp_img[:size_i,:size_i]
            if BS[:2] == "Pa":
                img[new_row - h_size_i+0: new_row + h_size_i+0, new_col - h_size_i:new_col + h_size_i] =\
                            Part_img[:size_i,:size_i]
            
            img = tools_img(tree[1][0], new_row, new_col, img )
            new_col = int(col+ cont_c *inc_col)
            
            params = tree[1][1][0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
                #RIGHT PHS
                img[new_row - h_size_i-30: new_row + h_size_i-30, new_col - h_size_i-37:new_col + h_size_i-37] =\
                            PHS_2_img[:size_i,:size_i]
                
            else:
                BS = params[0]
            ##cv2.putText(img, chain(BS),(new_col+15, new_row + 15), font, font_scale,(40), 2)
            
            if BS[:2] == "Su":
                img[new_row - h_size_i+0: new_row + h_size_i+0, new_col - h_size_i:new_col + h_size_i] =\
                            Superp_img[:size_i,:size_i]
            if BS[:2] == "Pa":
                img[new_row - h_size_i+0: new_row + h_size_i+0, new_col - h_size_i:new_col + h_size_i] =\
                            Part_img[:size_i,:size_i]

            img = tools_img(tree[1][1], new_row, new_col, img )
            
            cont_c *= delta_cont_c
            cont_r *= delta_cont_r
    return img

#Draw lines
def lines_img(tree, row, col, img):
    global cont_r, cont_c, font, font_scale
    if node_type(tree) == 'T':
        new_row = int(row + cont_r*inc_row)
        new_col = int(col - cont_c *inc_col)
    else:
        if len(tree[1]) == 1:
            BS = tree[1]
            new_row = int(row + cont_r*inc_row)
            new_col = int(col - cont_c *inc_col)

        elif len(tree[1]) == 2:  
            params = tree[0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
            else:
                BS = params[0]
                    
            new_row = int(row+ cont_r*inc_row)
            new_col = int(col- cont_c *inc_col)
            
            params = tree[1][0][0]
            params = params.split(" ")
            
            if len(params) > 1:
                PHS, BS = params
                
            else:
                BS = params[0]

      
            cv2.line(img, (col,row), (new_col, new_row), (180), 1)
            img = lines_img(tree[1][0], new_row, new_col, img )
            new_col = int(col+ cont_c *inc_col)
            

            params = tree[1][1][0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
                #RIGHT PHS

            else:
                BS = params[0]

            cv2.line(img, (col,row), (new_col, new_row), (180), 1)
            img = lines_img(tree[1][1], new_row, new_col, img )
            
            cont_c *= delta_cont_c
            cont_r *= delta_cont_r

    return img

# Draw text
def text_img(tree, row, col, img):
    global cont_r, cont_c, font, font_scale
    if node_type(tree) == 'T':
        new_row = int(row + cont_r*inc_row)
        new_col = int(col - cont_c *inc_col)
    else:
        if len(tree[1]) == 1:
            BS = tree[1]
            new_row = int(row + cont_r*inc_row)
            new_col = int(col - cont_c *inc_col)

        elif len(tree[1]) == 2:
            params = tree[0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
                ## <=  ## TEXTO RAIZ
                cv2.putText(img,chain(BS),(col+15, row  ), font, font_scale * 1.,(0), 2)
            else:
                BS = params[0]
                # <=
                cv2.putText(img,chain(BS),(col+15, row+15), font, font_scale,(0), 2)
                        
            new_row = int(row+ cont_r*inc_row)
            new_col = int(col- cont_c *inc_col)
            
            params = tree[1][0][0]
            params = params.split(" ")
            
            if len(params) > 1:
                PHS, BS = params
                #<=
                cv2.putText(img,chain(PHS),(new_col+50, new_row -25  ), font, font_scale * 1.,(0), 2)
                 
            else:
                BS = params[0]


            img = text_img(tree[1][0], new_row, new_col, img )
            new_col = int(col+ cont_c *inc_col)
            

            params = tree[1][1][0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params
                #RIGHT PHS
                cv2.putText(img, chain(PHS),(new_col-6, new_row - 25), font, font_scale * 1.,(40), 2)
                
            else:
                BS = params[0]
                
            img = text_img(tree[1][1], new_row, new_col, img )
            
            cont_c *= delta_cont_c
            cont_r *= delta_cont_r

    return img



c = \
['BS_0.31_83.04', [['BS_86.8_52.5', [['BS_43.22_15.87', [['BS_71.68_89.24', [['BS_44.14_2.0', [['Sup._42.2'], ['PHS_67.29 Sup._42.2']]], ['Sup._42.2']]], ['BS_45.21_1.17', [['Sup._42.2'], ['PHS_55.29 Sup._42.2']]]]], ['Part._0.95']]], ['Sup._42.2']]]

output_folder = './output_nsga/n_8/'
name = '_circuit_'
name += 'MU_100_GP_0.5_10_SOFT_2_1_best'



cont_r = 1; cont_c = 1;
inc_row = 70
inc_col = 70

delta_cont_c = 1.1 #1.13 #1.1 # 1.01 - 1.05     1.1
delta_cont_r = 1.3 # 1.42

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5 #0.5
row = 60; col = 300+ 50 + 30; #+ 100
img = 255 * np.ones((400+80,550+80)); #+100

tools_img(c, row, col, img); cont_r = 1; cont_c = 1;
lines_img(c, row, col, img); cont_r = 1; cont_c = 1;
text_img(c, row, col, img)

cv2.imwrite(output_folder + name + ".png", img)
print(expr(c))
