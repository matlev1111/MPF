from PIL import Image
import numpy as np
import math
from ESakCipher import *
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from modes import E_sak_CBC as Forward, Form_pic_blocks
import math

def Form_text_blocks(m, text_file):
    unicode_file = open(text_file)
    txt = unicode_file.read()
    strl = list()
    for c in txt:
        strl.append('{:02X}'.format(ord(c)))

    NM = []
    for i in range(len(strl)):
        sk = strl[i]
        NM.append(int(sk[0],base = 16))
        NM.append(int(sk[1], base = 16))


    diff = m**2 - (len(NM)-math.floor(len(NM)/(m**2))*(m**2))
    for i in range(diff):
        NM.append(0)
    k=-1
    print(len(NM))
    M1 = np.zeros([m,m,len(NM)//(m**2)])
    for i in range(0,len(NM),m**2):
        k += 1
        M1[:,:,k] = np.reshape(NM[i:i+m**2],[m,m])
    return M1.astype('int32')

print(Form_text_blocks(4, 'duom.txt'))
