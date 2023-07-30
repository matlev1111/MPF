from PIL import Image
import numpy as np
import math
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from M16 import *
import scipy.stats as stats
from modes import Form_pic_blocks, M16_CTR as Forward,Form_text_blocks

"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""

def M16CTR_F(m, file_name, save = False, rgb = 'L', nb = 8):
    """
    CTR mode of operation excryption
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
    #file_name = 'coloredChips_NoY'
    #file_name = 'linux'
    #Algorithm: M16, Esak
    file_folder = 'M16'
    #Mode: CTR, CBC
    mode = 'CTR'

    im = Image.open(f"images/{file_name}").convert(rgb)
    X1 = np.asarray(im)

    t = nb
    nbits = 2
    X = np.random.randint(0,np.power(2,t-1),size=(m,m))
    #Y2 = perm_matrix(m,t)
    Y2 = np.random.randint(0,np.power(2,t-1),size=(m,m))

    Ma = np.zeros([np.power(2,t),2])
    for a in range(np.power(2,t)):
        Ma[a,0] = math.floor(a/np.power(2,t-1))
        Ma[a,1] = np.mod(a,np.power(2,t-1))
    if(rgb == 'L'):
        prgb = 1
    else:
        prgb = 3
    s1, s2, M1 = Form_pic_blocks(m, X1,prgb, nb = nb)
    #M1 = Form_text_blocks(m, 'Generated.txt', nb = nb)
    #M = FormM(M1, m, Ma, 1)
    delta = np.zeros([m,m,2]).astype('int32')
    delta[:,:,0] = np.random.randint(0,2,size=(m,m))

    Nblocks = M1.shape[2]
    start = timer()
    C = Forward(M1, Ma, X, Y2, Nblocks, t, nbits, delta)
    end = timer()
    #print(timedelta(seconds=end-start))
    #unique, counts = np.unique(np.array(C), return_counts=True)
    #print(stats.chisquare(f_obs=counts, f_exp=np.ones(np.power(2,t), dtype = 'int32')*(Nblocks*m*m/np.power(2,t))))
    if(save):
        file = open(f'res/{mode}/{file_folder}/{file_name}_4x4_NoY_addmod_cntr7', 'wb')
        pickle.dump([C, M1, s1, s2], file)
        #pickle.dump([C, M1], file)
        file.close()
    return end-start

sk = 100
for i in range(sk):
    tmp = M16CTR_F(4, 'Generated8',save = False, nb = 8)
    if(i == 0):
        CC = np.zeros([tmp.shape[0],tmp.shape[1],tmp.shape[2],sk])
    CC[:,:,:,i] = tmp[:,:,:]

file = open(f'res/CTR/M16/Full_text_C', 'wb')
pickle.dump(CC, file)
file.close()

#M16CTR_F(4, 'Generated8',save = True, nb = 8)
