import numpy as np
from ESakCipher import *
from M16 import *


def E_sak_CBC(IV,M, G, X, Y, rounds, p,q):
    m = X.shape[0]
    Fmap = Gmap(G,X)
    CF = np.zeros([m,m,rounds+1], dtype='int32')
    CF[:,:,0] = IV
    for i in range(rounds):
        """if(i % 100 == 0):
            print(i/rounds*100)"""
        M1 = np.mod(np.bitwise_xor(CF[:,:,i],M[:,:,i]),q)
        C = ESakCipher(G,X,M1,Y,p,q, Fmap)
        CF[:,:,i+1] = C
    return CF

def E_sak_CTR(M, G, X, Y, rounds, p,q):
    m = X.shape[0]
    Fmap = Gmap(G,X)
    Cnt = np.random.randint(1,q,size=(m,m)).astype('int32')
    """Cnt = np.matrix([
   [487,   455,   365,   283],
   [401,   181,   263,   341],
   [487,   219,   243,     8],
     [0,   320,     0,     0]
    ])"""
    Cnt[m-1,m-1] = 1
    Cnt[m-1,m-2] = 1
    Cnt[m-1,m-3] = 1
    Cnt[m-1,m-4] = 1
    CF = np.zeros([m,m,rounds])
    for i in range(rounds):
        """if(i % 100 == 0):
            print(i/rounds*100)"""
        M1 = Cnt.copy()
        Cnt[m-1,m-1] += 1
        pind = m-1
        while(Cnt[m-1,pind] == q):
            Cnt[m-1,pind] = 1
            Cnt[m-1,pind-1] += 1
            pind -= 1
        C = ESakCipher(G,X,M1,Y,p,q, Fmap)
        """print("C", C)
        print("M", M[:,:,i]+255)"""
        CF[:,:,i] = np.mod(C+M[:,:,i],q)#np.mod(np.bitwise_xor(C,M[:,:,i]),q)
        #print("CF", CF[:,:,i])
    return CF.astype('int32')

def Form_pic_blocks(m, X1, rgb):
    ss = X1.shape
    #print(ss)
    if(ss[0]%m != 0):
        s1 = (ss[0]//m+1)*m
    else:
        s1 = ss[0]//m*m
    if(ss[1]%m != 0):
        s2 = (ss[1]//m+1)*m
    else:
        s2 = ss[1]//m*m
    #print(s1,s2)
    if(rgb == 3):
        ZZ = np.zeros([s1, s2, rgb])
        ZZ[:ss[0],:ss[1],:] = X1
    else:
        ZZ = np.zeros([s1, s2])
        ZZ[:ss[0],:ss[1]] = X1
    finish = s1*s2//(m*m)
    M1 = np.zeros([m,m,finish*rgb])
    indx = 0
    for k in range(rgb):
        for i in range(s1//m):
            for j in range(s2//m):
                if(rgb ==3):
                    M1[:,:,indx] = ZZ[i*m:(i+1)*m, j*m:(j+1)*m,k]
                else:
                    M1[:,:,indx] = ZZ[i*m:(i+1)*m, j*m:(j+1)*m]
                indx += 1
    return s1, s2, M1.astype('int32')

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
    M1 = np.zeros([m,m,len(NM)//(m**2)])
    for i in range(0,len(NM),m**2):
        k += 1
        M1[:,:,k] = np.reshape(NM[i:i+m**2],[m,m])
    return M1.astype('int32')


def M16_CTR(M, Ma, X, Y2, rounds, t, nbits, delta):
    m = X.shape[0]
    Cnt = np.random.randint(0,np.power(2,t),size=(m,m,1)).astype('int32')
    Cnt[m-1,m-1,0] = 0
    Cnt[m-1,m-2,0] = 0
    Cnt[m-1,m-3,0] = 0
    Cnt[m-1,m-4,0] = 0
    CF = np.zeros([m,m,rounds])
    for i in range(rounds):
        if(i % 100 == 0):
            print(i/rounds*100)
        M1 = FormM(Cnt, m, Ma, 1)
        Cnt[m-1,m-1,0] += 1
        pind = m-1
        while(Cnt[m-1,pind] == np.power(2,t)):
            Cnt[m-1,pind,0] = 0
            Cnt[m-1,pind-1,0] += 1
            pind -= 1
        C = M16_Enc(M1[:,:,0,:],delta, X,Y2,t, nbits)
        CF[:,:,i] = np.mod(C+M[:,:,i],np.power(2,t))#np.mod(np.bitwise_xor(C,M[:,:,i]),np.power(2,t))
    return CF

def M16_CBC(IV,M, Ma, X, Y2, rounds, t, nbits, delta):
    m = X.shape[0]
    CF = np.zeros([m,m,rounds+1], dtype='int32')
    CF[:,:,0] = IV
    for i in range(rounds):
        if(i % 100 == 0):
            print(i/rounds*100)
        M1 = np.mod(np.bitwise_xor(CF[:,:,i],M[:,:,i]),np.power(2,t))
        M2 = FormM(M1.reshape([m,m,1]), m, Ma, 1)
        C = M16_Enc(M2[:,:,0,:],delta, X,Y2,t, nbits)
        CF[:,:,i+1] = C
    return CF