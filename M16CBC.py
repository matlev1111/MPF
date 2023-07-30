from PIL import Image
import numpy as np
import math
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from M16 import *
import scipy.stats as stats
from modes import Form_pic_blocks, M16_CBC as Forward,Form_text_blocks

"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""

def M16CBC_F(m, file_name, save = False, rgb = 'L', nb = 8):
    #file_name = 'linux'
    #Algorithm: M16, Esak
    file_folder = 'M16'
    #Mode: CTR, CBC
    mode = 'CBC'

    im = Image.open(f"images/{file_name}").convert(rgb)#Image.open("coloredChips.png")
    X1 = np.asarray(im)
    
    t = nb
    nbits = 2
    X = np.random.randint(0,np.power(2,t-1),size=(m,m))
    Y2 = perm_matrix(m,t)

    Ma = np.zeros([np.power(2,t),2])
    for a in range(np.power(2,t)):
        Ma[a,0] = math.floor(a/np.power(2,t-1))
        Ma[a,1] = np.mod(a,np.power(2,t-1))
    if(rgb == 'L'):
        prgb = 1
    else:
        prgb = 3
    start = timer()
    s1, s2, M1 = Form_pic_blocks(m, X1,prgb, nb = nb)
    print("Generavimo laikai - ",timer()-start)
    #M1 = Form_text_blocks(m, 'Generated.txt', nb = nb)
    #M1 = FormM(M1, m, Ma, 1)
    delta = np.zeros([m,m,2]).astype('int64')
    delta[:,:,0] = np.random.randint(0,2,size=(m,m))

    Nblocks = M1.shape[2]
    IV = np.zeros([m,m], dtype=int)
    start = timer()
    C = Forward(IV,M1, Ma, X, Y2, Nblocks, t, nbits, delta)
    end = timer()
    #print(timedelta(seconds=end-start))

    #unique, counts = np.unique(np.array(C[:,:,1:]), return_counts=True)
    #print(stats.chisquare(f_obs=counts, f_exp=np.ones(np.power(2,t), dtype = 'int64')*(Nblocks*m*m/np.power(2,t))))
    if(save):
        file = open(f'res/{mode}/{file_folder}/{file_name}_4x4_addmod', 'wb')
        pickle.dump([C, M1, s1, s2], file)
        #pickle.dump([C, M1], file)
        file.close()
    return end-start

#M16CBC_F(4, 'cameraman.tif',save = False, nb = 16)
