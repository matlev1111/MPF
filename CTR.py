from PIL import Image
import numpy as np
import math
from ESakCipher import *
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from modes import E_sak_CTR as Forward, Form_pic_blocks 

"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""


def CTR_F(m):
    file_name = 'coloredChips'
    #Algorithm: M16, Esak
    file_folder = 'Esak'
    #Mode: CTR, CBC
    mode = 'CTR'
    im = Image.open("images/coloredChips.png").convert('RGB')#Image.open("coloredChips.png")
    X1 = np.asarray(im)


    #m = 4
    p = 563
    q, G, X, Y, Z = Gen_parameters(m,p)
    """X = np.matrix([

        [95,    13,   205,   194],
        [82,   187,   198,   156],
    [209,   169,   219,   111],
        [2,   147,    80,    17]
    ])

    Y = np.matrix([
    [219,    30,   250,    21],
        [95,    36,   224,    25],
    [171,   154,   206,   224],
    [208,   136,    15,   265]
    ])
    Z = np.matrix([
    [192,    33,   210,   206],
        [37,   180,   163,   272],
    [203,    92,   207,   243],
        [31,   183,    65,    24]
    ])"""
    s1, s2, M1 = Form_pic_blocks(m, X1,3)

    Nblocks = M1.shape[2]

    start = timer()
    C = Forward(M1, G, X, Y, Nblocks, p, q)
    end = timer()
    print(timedelta(seconds=end-start))
    return end-start
TM = 0
FT = []

for m in range(3,11):
    for i in range(5):
        print(f"m = {m}")
        TM += CTR_F(m)
    FT.append(TM/5)


file = open(f'res/CTR_times', 'wb')
pickle.dump([FT], file)
file.close()
for i in range(len(FT)):
    print(f"m = {i+3}, ",timedelta(seconds=FT[i]))
"""file = open(f'res/{mode}/{file_folder}/{file_name}', 'wb')
pickle.dump([C, M1, s1, s2], file)
file.close()"""
