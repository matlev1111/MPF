from PIL import Image
import numpy as np
import math
from ESakCipher import *
from timeit import default_timer as timer
from datetime import timedelta
import pickle
import scipy.stats as stats
from modes import E_sak_CBC as Forward, Form_pic_blocks, Form_text_blocks

def CBC_F(m, file_name, save = False, rgb = 'L', nb = 8):
    """
    CBC mode of operation excryption
    Commented parts are for text
    Input:
        m - matrix order
        file_name - file name
        save - True to save, False not to save
        rgb - 'L' grayscale, 'RGB' - RGB
        nb - number of bits, default 8bits
    Output:
        Time how long the encryption lasted, if save = True, ciphertext is saved as pickle file
    """
    #file_name = 'linux'
    #Algorithm: M16, Esak
    file_folder = 'Esak'
    #Mode: CTR, CBC
    mode = 'CBC'
    im = Image.open(f"images/{file_name}").convert(rgb)
    X1 = np.asarray(im)
    if(nb == 4):
        p = 47
    elif (nb == 8):
        p = 563
    elif (nb == 16):
        p = 131267
    if(rgb == 'L'):
        prgb = 1
    else:
        prgb = 3
    q, G, X, Y, Z = Gen_parameters(m,p)
    start = timer()
    s1, s2, M1 = Form_pic_blocks(m, X1,prgb, nb=nb)
    print("Generavimo laikai - ",timer()-start)
    #For text encryption
    #M1 = Form_text_blocks(m, 'Generated.txt', nb = nb)

    IV = np.zeros([m,m], dtype=int)
    Nblocks = M1.shape[2]

    start = timer()
    C = Forward(IV,M1, G, X,Y,Z,  Nblocks, p, q)
    end = timer()
    print(timedelta(seconds=end-start))
    unique, counts = np.unique(np.array(C[:,:,1:]), return_counts=True)
    q = int(q)
    print(stats.chisquare(f_obs=counts, f_exp=np.ones(q, dtype = 'int64')*(Nblocks*m*m/q)))
    if(save):
        file = open(f'res/{mode}/{file_folder}/{file_name}_4x4_addmod', 'wb')
        pickle.dump([C, M1, s1, s2], file)
        #pickle.dump([C, M1], file)
        file.close()
    return end-start


CBC_F(4, 'cameraman.tif',save = False, nb = 16)




