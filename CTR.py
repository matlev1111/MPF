from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import math
from ESakCipher import *
from timeit import default_timer as timer
from datetime import timedelta
import pickle
"""
im = Image.open("coloredChips.png").convert('L')
image = np.asarray(im)
Image.fromarray(image).show()
im.show()
"""

def Forward(M, G, X, ind,rounds, p,q):
    m = X.shape[0]
    Fmap = Gmap(G,X)
    Cnt = np.random.randint(1,q,size=(m,m)).astype('int32')
    Cnt[m-1,m-1] = 0
    Cnt[m-1,m-2] = 0
    Cnt[m-1,m-3] = 0
    Cnt[m-1,m-4] = 0
    CF = np.zeros([m,m,rounds])
    for i in range(rounds):
        if(i % 100 == 0):
            print(i/rounds*100)
        M1 = Cnt
        Cnt[m-1,m-1] += 1
        pind = m-1
        while(Cnt[m-1,pind] == q):
            Cnt[m-1,pind] = 0
            Cnt[m-1,pind-1] += 1
            pind -= 1
        C = ESakCipher(G,X,M1,Y,p,q, Fmap)
        CF[:,:,i] = np.mod(imap(ind, C,p)+M[:,:,i],q)
    return CF


im = Image.open("images/coloredChips.png").convert('RGB')#Image.open("coloredChips.png")
X1 = np.asarray(im)


m = 4
p = 563
q = (p-1)/2
bitsq = math.ceil(np.log(q)/np.log(2))
ind, G = GenG(p)
X = np.random.randint(0,q,size=(m,m))
Y = np.random.randint(1,q,size=(m,m))
Z = np.random.randint(0,q,size=(m,m))

ss = X1.shape
print(ss)
if(ss[0]%m != 0):
    s1 = (ss[0]//m+1)*m
else:
    s1 = ss[0]//m*m
if(ss[1]%m != 0):
    s2 = (ss[1]//m+1)*m
else:
    s2 = ss[1]//m*m
print(s1,s2)
rgb = 3
ZZ = np.zeros([s1, s2, rgb])
ZZ[:ss[0],:ss[1],:] = X1
finish = s1*s2//(m*m)
M1 = np.zeros([m,m,finish*rgb])
indx = 0
for k in range(rgb):
    for i in range(s1//m):
        for j in range(s2//m):
            M1[:,:,indx] = ZZ[i*m:(i+1)*m, j*m:(j+1)*m,k]
            indx += 1

while(np.mod(np.linalg.det(Y),q)==0):
    Y = np.random.randint(1,q,size=(m,m))

IV = np.zeros([m,m])
Nblocks = M1.shape[2]
print(Nblocks)
bitsS = math.ceil(np.log(Nblocks)/np.log(2))

start = timer()
C = Forward(M1, G, X, ind, Nblocks, p, q)
end = timer()
print(timedelta(seconds=end-start))

file = open('res/res1', 'wb')
pickle.dump([C, s1, s2], file)
file.close()
