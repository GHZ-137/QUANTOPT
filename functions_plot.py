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

def plot(values, x_values, labels, err = [], y_inf = 0., y_sup = 1., d_x = 2, grid = True, grey = False, x_label ='', y_label ='', title = '', f_name = 'image.png', points = False, loc="upper right"):
    """
    Make a plot of lines of values and saves it to an image.
    Arguments:
        values - list of arrays of values
        x_values - array of independent variable values
        labels - list of labels of each array
        y_inf, y_sup - min. and max. of y axis
        d_x - spacing between x axis ticks
        grid - whether overimpose a dashed grid or not (default yes)
        f_name - image file name
    """
    colors = ['blue', 'green', 'orange']
    x_max = len(values[0])

    plt.clf()
    ax = plt.figure().gca()

    x_ticks = list( range(0, len(x_values)+1, d_x) )
    print(x_ticks)
 
    x_labels = [0] + ["%.d" % (x_values[idx-1]+1) for idx in x_ticks[1:]]
    ax.set_xticks( x_ticks, minor = False)
    ax.set_xticklabels(x_labels, rotation = 90)
    

    if grid:
        ax.xaxis.grid(linestyle= 'dashed')
        ax.yaxis.grid(linestyle= 'dashed')
    plt.ylim( [y_inf, y_sup] )
    plt.xlim( left=0 )

    if x_label != '':
        plt.xlabel(x_label)
    if y_label != '':
        plt.ylabel(y_label)        

    for cont in range(len(values)):
        color = colors[ cont % len(colors)]
        if grey and cont ==0:
            color = 'blue' 
        if grey and cont !=0:
            color = (0.,0.,1., .2)

        if cont < len(labels):
            label = labels[cont]
        else:
            label = None

        if points:
            plt.plot(range(0, x_max), values[cont], label = label, linestyle='--', marker='o', color = color )
        else:
            plt.plot(range(0, x_max), values[cont], label = label, color = color )
        #print( err[cont] )   
        if err != []:
            #plt.errorbar(range(0, x_max), values[cont], err[cont], ecolor = 'grey', barsabove = True, errorevery = d_x, label = label,  color = color )
    
             
            plt.fill_between(range(0, x_max), values[cont]- err[cont], values[cont] + err[cont], color = color, alpha = 0.25)
    plt.legend(loc = loc)
    if title !='':
        plt.title(title)
    plt.tight_layout()
    
    print (f_name)
    plt.savefig(f_name)
    return

def plot_multi(values, x_values, labels, y_inf = 0., y_sup = 1., d_x = 2, grid = True, grey = False, x_label ='', y_label ='', title = '', f_name = 'image.png', points = False):
    """
    Make a plot of lines of values and saves it to an image.
    Arguments:
        values - list of list of arrays of values
        x_values - array of independent variable values
        labels - list of labels of each list
        y_inf, y_sup - min. and max. of y axis
        d_x - spacing between x axis ticks
        grid - whether overimpose a dashed grid or not (default yes)
        f_name - image file name
    """
    colors = ['blue', 'green', 'orange']
    x_max = len(values[0][0])

    plt.clf()
    ax = plt.figure().gca()

    x_ticks = list( range(0, len(x_values)+1, d_x) )
    print(x_ticks)
    x_labels = [0] + ["%.d" % (x_values[idx-1]+1) for idx in x_ticks[1:]]
    ax.set_xticks( x_ticks, minor = False)
    ax.set_xticklabels(x_labels, rotation = 90)
    

    if grid:
        ax.xaxis.grid(linestyle= 'dashed')
        ax.yaxis.grid(linestyle= 'dashed')
    plt.ylim( [y_inf, y_sup] )
    plt.xlim( left=0 )

    if x_label != '':
        plt.xlabel(x_label)
    if y_label != '':
        plt.ylabel(y_label)        


    for cont1 in range(len(values)):
        color = colors[ cont1 % len(colors)]
        if grey:
            color = (0.5,0.5,0.5, .2)
        
        for cont2 in range(len(values[cont1])):
            if cont2 == 0:
                label = labels[cont1]
            else:
                label = None
            
            al = .6 #0.5 + 0.5 * (float(cont2) / len(values[cont1]))
        
            if points:
                plt.plot(range(0, x_max), values[cont1][cont2], label = label, linestyle='--', marker='o', color = color, alpha = al )
            else:
                print(values[cont1][cont2].shape)
                plt.plot(range(0, x_max), values[cont1][cont2], label = label, color = color, alpha = al )


      
    plt.legend(loc="lower right")
    if title !='':
        plt.title(title)
    plt.tight_layout()
    
    print (f_name)
    plt.savefig(f_name)
    return


def scatter(values, x_values, labels, y_inf = 0., y_sup = 1., d_x = 2, grid = True, grey = False, x_label ='', y_label ='', f_name = 'image.png'):
    """
    Make a plot of lines of values and saves it to an image.
    Arguments:
        values - list of arrays of values
        x_values - array of independent variable values
        labels - list of labels of each array
        y_inf, y_sup - min. and max. of y axis
        d_x - spacing between x axis ticks
        grid - whether overimpose a dashed grid or not (default yes)
        f_name - image file name
    """
    colors = ['blue', 'green', 'orange']
    x_max = len(values[0])

    plt.clf()
    ax = plt.figure().gca()

    x_ticks = range(0, len(x_values), d_x)
    x_labels = ["%.2f" % x_values[idx] for idx in x_ticks]
    ax.set_xticks( x_ticks, minor = False)
    ax.set_xticklabels(x_labels, rotation = 90)
    

    if grid:
        ax.xaxis.grid(linestyle= 'dashed')
        ax.yaxis.grid(linestyle= 'dashed')
    plt.ylim( [y_inf, y_sup] )
    plt.xlim( left=0 )

    if x_label != '':
        plt.xlabel(x_label)
    if y_label != '':
        plt.ylabel(y_label)        

    for cont in range(len(values)):
        color = colors[ cont % len(colors)]
        if grey and cont ==0:
            color = 'blue' 
        if grey and cont !=0:
            color = (0.,0.,1., .2)

        if cont < len(labels):
            label = labels[cont]
        else:
            label = None
            s
        plt.plot(range(0, x_max),values[cont], label = label,  linestyle='--', marker='o', color = color )
    plt.legend(loc="lower right")

    print (f_name)
    plt.savefig(f_name)
    return



def plt_grid(f_name, grid, title = ''):
    """ Save grid plot of a 2D array """
    plt.clf()
    h, w = grid.shape
    m = np.min([w,h])

    ##plt.bar3d(x, y, bottom, width, depth, top, shade=True)
    
    #plt.figure(figsize = ((h/m) * 5., (w/m) * 5.))
    plt.pcolor(grid, cmap = 'Spectral_r', edgecolors = 'k')
    plt.colorbar()
    ###plt.clim(-0.01, None)
    ##plt.clim(None, 0.15)
    #plt.colorbar(fraction=0.04, pad=0.04)
    plt.title(title)
    plt.savefig(f_name)
    
    return

def plt_bar3d(f_name, grid, title, z_lim):
    """ Save 3d bar plot of a 2D array """
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_zlim(0., z_lim)
    h, w = grid.shape

    y = range(h)
    x = range(w)
    z = grid.flatten()
    xx, yy = np.meshgrid(x, y)
    X, Y = xx.ravel(), yy.ravel()
    height = np.zeros_like (z)
    

    dx = .5
    dy = .5
    dz = z
    ax1.bar3d(X, Y, height, dx, dy, dz, color='#00ceaa', shade=True)
    #ax1.set_xlabel('X')

    plt.title(title)
    plt.savefig(f_name)
    
    return

# List of tree coord. and parameters
def draw_params(params, row, col, img):
    global inc_row, inc_col, font_scale
    params = params.split(" ")
    if len(params) > 1:
        PHS, BS = params   
        cv2.putText(img,PHS,(col, row - inc_row//2), font, font_scale * 1.,(0), 2)
    else:
        BS = params[0]
    #col += inc_col
    #row += inc_row
    cv2.putText(img,BS,(col, row), font, font_scale,(0), 2)
        
    return img


def tree_coord(tree, row, col):
    global inc_row, inc_col
    if node_type(tree) == 'T':
        row += inc_row
        res = [row + inc_row, col + inc_col, str( tree[0] )]
    else:
        if len(tree[1]) == 1:
            res = [row + inc_row, col + inc_col, str( tree)]
        elif len(tree[1]) == 2:
            #res = '<' + expr(tree[1][0], dec) +'|'+  str(tree[0]) +'|' + expr(tree[1][1], dec)+'>'
            
            res = [[row + inc_row, col + inc_col, str( tree[0] )]]
            res.append( tree_coord(tree[1][0], row + inc_row, col - inc_col) )
            res.append( tree_coord(tree[1][1], row + inc_row, col + inc_col) )
    
    return res

def chain(l):
    l = l.split('_')
    res = l[0] + '(' + l[1]
    if len(l) == 3:
        res += ', '+ l[2]+')'
    else:
        res += ')'
    return res

# Create a tree circuit image
def tree_img(tree, row, col, img):
    global cont_r, cont_c, font, font_scale
    if node_type(tree) == 'T':
        #print(len(tree))
        new_row = int(row+ cont_r*inc_row)
        new_col = int(col- cont_c *inc_col)
        #row += inc_row
        ###img = draw_params(tree[0],new_row, new_col, img )
    else:
        if len(tree[1]) == 1:
            #img = draw_params(tree, row, col , img )
            BS = tree[1]
            new_row = int(row+ cont_r*inc_row)
            new_col = int(col- cont_c *inc_col)
            cv2.putText(img, chain(BS),(new_col+15, new_row+15), font, font_scale,(40), 2)
            
        elif len(tree[1]) == 2:
            #res = '<' + expr(tree[1][0], dec) +'|'+  str(tree[0]) +'|' + expr(tree[1][1], dec)+'>'

            ##img = draw_params(tree[0], row, col, img )
            
            params = tree[0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params   
                cv2.putText(img,chain(PHS),(col+15, row  ), font, font_scale * 1.,(0), 2)
            else:
                BS = params[0]
            cv2.putText(img,chain(BS),(col+15, row +15), font, font_scale,(40), 2)
            cv2.circle(img, (col, row), 7, (180), - 1)
            
            ###img = tree_img(tree[1][0], new_row, new_col, img )
            
            new_row = int(row+ cont_r*inc_row)
            new_col = int(col- cont_c *inc_col)
            cv2.line(img, (col,row), (new_col, new_row), (180), 1)
            
            params = tree[1][0][0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params   
                cv2.putText(img,chain(PHS),(new_col+15, new_row  ), font, font_scale * 1.,(0), 2)
            else:
                BS = params[0]
            cv2.putText(img, chain(BS),(new_col+15, new_row +15), font, font_scale,(40), 2)
            cv2.circle(img, (new_col, new_row), 7, (180), -1)

            img = tree_img(tree[1][0], new_row, new_col, img )
            
            
            new_col = int(col+ cont_c *inc_col)
            cv2.line(img, (col,row), (new_col, new_row), (180), 1)

            params = tree[1][1][0]
            params = params.split(" ")
            if len(params) > 1:
                PHS, BS = params   
                cv2.putText(img, chain(PHS),(new_col+15, new_row), font, font_scale * 1.,(40), 2)
            else:
                BS = params[0]
            cv2.putText(img, chain(BS),(new_col+15, new_row + 15), font, font_scale,(40), 2)
            cv2.circle(img, (new_col, new_row), 7, (180), -1)
            
            img = tree_img(tree[1][1], new_row, new_col, img )
            
            cont_c *= 1.0
            cont_r *= 1.5
    
       
    
    return img

"""
c = ['BS_78.38_52.5', [['BS_42.44_0.0', [['BS_54.18_77.13', [['PHS_2.6 BS_43.58_0.0', [['Sup._42.2'], ['Sup._42.2']]], ['BS_59.81_9.14', [['PHS_59.02 Sup._42.2'], ['Sup._42.2']]]]], ['BS_44.51_0.0', [['Sup._42.2'], ['Sup._42.2']]]]], ['Sup._42.2']]]
c = ['BS_45_0', [['BS_45_0', [['BS_45_0', [['PHS_25 Part._0.95'], ['Sup._42.2']]], ['Part._0.95']]], ['Sup._42.2']]]

cont_r = 1; cont_c = 1;
inc_row = 70
inc_col = 70
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
row = 60; col = 300;
img = 255 * np.ones((600,900))
#c = ['BS_7.2_0.0', [['BS_44.94_0.0', [['BS_51.19_6.45', [['PHS_8.41 Sup._42.2'], ['Sup._42.2']]], ['BS_46.98_0.0', [['PHS_3.71 Sup._42.2'], ['PHS_80.52 Sup._42.2']]]]], ['PHS_60.22 Sup._42.2']]]

c_coord = tree_coord(c, row, col)
#print(c_coord)
tree_img(c, row, col, img)
cv2.imwrite("results/Ejemplo_.png", img)
print(expr(c))
"""
