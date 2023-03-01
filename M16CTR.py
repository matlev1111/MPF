from PIL import Image
import numpy as np
import math
from timeit import default_timer as timer
from datetime import timedelta
import pickle
from M16 import *
from modes import Form_pic_blocks, M16_CTR as Forward

"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""
file_name = 'coloredChips_NoY'
#Algorithm: M16, Esak
file_folder = 'M16'
#Mode: CTR, CBC
mode = 'CTR'

im = Image.open("images/coloredChips.png").convert('RGB')#Image.open("coloredChips.png")
X1 = np.asarray(im)

t = 8
m = 4
nbits = 2
X = np.random.randint(0,np.power(2,t-1),size=(m,m))
#Y2 = perm_matrix(m,t)
Y2 = np.random.randint(0,np.power(2,t-1),size=(m,m))

Ma = np.zeros([256,2])
for a in range(256):
    Ma[a,0] = math.floor(a/np.power(2,t-1))
    Ma[a,1] = np.mod(a,np.power(2,t-1))

s1, s2, M1 = Form_pic_blocks(m, X1)
print(M1.shape)
M = FormM(M1, m, Ma, 1)
delta = np.zeros([m,m,2]).astype('int32')
delta[:,:,0] = np.random.randint(0,2,size=(m,m))

Nblocks = M1.shape[2]
print(Nblocks)
start = timer()
C = Forward(M1, Ma, X, Y2, Nblocks, t, nbits, delta)
end = timer()
print(timedelta(seconds=end-start))

file = open(f'res/{mode}/{file_folder}/{file_name}', 'wb')
pickle.dump([C, M1, s1, s2], file)
file.close()
