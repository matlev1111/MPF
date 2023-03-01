from PIL import Image
import numpy as np
import math
from ESakCipher import *
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from modes import E_sak_CBC as Forward, Form_pic_blocks, Form_text_blocks
"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""

def CBC_F(m):
    file_name = 'coloredChips'
    #Algorithm: M16, Esak
    file_folder = 'Esak'
    #Mode: CTR, CBC
    mode = 'CBC'
    im = Image.open("images/coloredChips.png").convert('RGB')#Image.open("coloredChips.png")
    X1 = np.asarray(im)
    #m = 4
    p = 563
    q, G, X, Y, Z = Gen_parameters(m,p)
    s1, s2, M1 = Form_pic_blocks(m, X1,3)

    """M1 = Form_text_blocks(m, 'duom.txt')"""

    IV = np.zeros([m,m], dtype=int)
    Nblocks = M1.shape[2]
    #print(Nblocks)

    start = timer()
    C = Forward(IV,M1, G, X,Y,  Nblocks, p, q)
    end = timer()
    print(timedelta(seconds=end-start))
    return end-start

TM = 0
FT = []

for m in range(3,11):
    for i in range(5):
        print(f"m = {m}")
        TM += CBC_F(m)
    FT.append(TM/5)


file = open(f'res/CBC_times', 'wb')
pickle.dump([FT], file)
file.close()
for i in range(len(FT)):
    print(f"m = {i+3}, ",timedelta(seconds=FT[i]))
"""file = open(f'res/{mode}/{file_folder}/{file_name}', 'wb')
pickle.dump([C, M1, s1, s2], file)
file.close()"""
