import numpy as np
from ESakCipher import *
from M16 import *


def E_sak_CBC(IV,M, G, X, Y,Z, rounds, p,q):
    """
    CBC mode used with algorithm based on commutative group 
    Inputs:
        IV - vector of inicialization
        M - plaintext
        G - list of indexes for mapping
        X - private key matrix
        Y - private key matrix
        Z - private key matrix
        rounds - number of rounds
        p - group order
        q - Syllow subgroup order
    Output:
        CF - ciphertext of all rounds
    """
    m = X.shape[0]
    Fmap = Gmap(G,Z)
    CF = np.zeros([m,m,rounds+1], dtype='int64')
    CF[:,:,0] = IV
    for i in range(rounds):
        M1 = np.mod(CF[:,:,i]+M[:,:,i],q)#np.mod(CF[:,:,i]+M[:,:,i],q)#np.mod(np.bitwise_xor(CF[:,:,i],M[:,:,i]),q)
        C = ESakCipher(G,X,M1,Y,p,q, Fmap)
        CF[:,:,i+1] = C
    return CF

def E_sak_CTR(M, G, X, Y, Z, rounds, p,q):
    """
    CTR mode used with algorithm based on commutative group
    Inputs:
        M - plaintext
        G - list of indexes for mapping
        X - private key matrix
        Y - private key matrix
        Z - private key matrix
        rounds - number of rounds
        p - group order
        q - Syllow subgroup order
    Output:
        CF - ciphertext of all rounds
    """
    m = X.shape[0]
    Fmap = Gmap(G,Z)
    Cnt = np.random.randint(1,q,size=(m,m)).astype('int32')
    Cnt[m-1,m-1] = 0
    Cnt[m-1,m-2] = 0
    Cnt[m-1,m-3] = 0
    Cnt[m-1,m-4] = 0
    CF = np.zeros([m,m,rounds])
    for i in range(rounds):
        M1 = Cnt.copy()
        Cnt[m-1,m-1] += 1
        pind = m-1
        while(Cnt[m-1,pind] >= q):
            Cnt[m-1,pind] = 0
            Cnt[m-1,pind-1] += 1
            pind -= 1
        C = ESakCipher(G,X,M1,Y,p,q, Fmap)
        CF[:,:,i] = np.mod(C+M[:,:,i],q)#np.mod(np.bitwise_xor(C,M[:,:,i]),q)
    return CF.astype('int32')

def merg2mat(M1, M2):
    """
    Merging two matrices into one, making 16-bit element
    Inputs:
        M1 - First matrix
        M2 - second matrix
    Output:
     NM - merged matrix
    """
    m = len(M1)
    NM = np.zeros(M1.shape)
    for i in range(m):
        for j in range(m):
            NM[i,j] = int('{:02X}'.format(int(M1[i,j]))+ '{:02X}'.format(int(M2[i,j])), base=16)
    return NM

def spl2mat(M):
    """
    Split matrix into 2, 4-bit elements
    Inputs:
        M - initial matrix for splitting
    Outputs:
        M1 - First matrix
        M2 - second matrix
    """
    M1 = np.zeros(M.shape)
    M2 = np.zeros(M.shape)
    for i in range(len(M)):
        for j in range(len(M)):
            sk = '{:02X}'.format(int(M[i,j]))
            M1[i,j] = int(sk[0],base = 16)
            M2[i,j] = int(sk[1], base = 16)
    return M1, M2

def Form_pic_blocks(m, X1, rgb, nb=8):
    """
    Making block from Picture
    Inputs:
        m - matrix order
        X1 - Initial picture
        rgb - format 1 - grayscale, 3 - rgb
        nb - number of bits
    Output:
        s1 - number of rows
        s2 - number or columns
        M1 - New matrix
    """
    ss = X1.shape
    if(ss[0]%m != 0):
        s1 = (ss[0]//m+1)*m
    else:
        s1 = ss[0]//m*m
    if(ss[1]%m != 0):
        s2 = (ss[1]//m+1)*m
    else:
        s2 = ss[1]//m*m
    if(rgb == 3):
        ZZ = np.zeros([s1, s2, rgb])
        ZZ[:ss[0],:ss[1],:] = X1
    else:
        ZZ = np.zeros([s1, s2])
        ZZ[:ss[0],:ss[1]] = X1
    finish = (s1//m)*(s2//m)

    if(finish*rgb%2 != 0 and nb == 16):
        M1 = np.zeros([m,m,finish*rgb+1])
    else:
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
    if(nb==16):
        if(indx%2 != 0):
            if(rgb ==3):
                M1[:,:,indx] = np.zeros([m,m,1])
            else:
                M1[:,:,indx] = np.zeros([m,m])
            indx += 1
        MM = np.zeros([m,m,(indx)//2])
        indx2 = 0
        for i in range(0,indx,2):
            MM[:,:,indx2] = merg2mat(M1[:,:,i],M1[:,:,i+1])
            indx2 += 1 
        M1 = MM
    elif(nb==4):
        MM = np.zeros([m,m,(indx+1)*2])
        indx2 = 0
        for i in range(0,indx):
            MM[:,:,indx2],MM[:,:,indx2+1] = spl2mat(M1[:,:,i])
            indx2 += 2 
        M1 = MM
    
    return s1, s2, M1.astype('int32')

def Form_text_blocks(m, text_file, nb = 8):
    """
    Making blocks from text
    Inputs:
        m - matrix order
        text_file - name of the text file
        nb - number of bits
    Output:
        M1 - new formed matrix
    """
    unicode_file = open(text_file,encoding='utf-8')
    txt = unicode_file.read()
    strl = list()
    for c in txt:
        strl.append('{:02X}'.format(ord(c)))
    NM = []
    #Splits 8bit elements into 2 4bit elements
    for i in range(len(strl)):
        sk = strl[i]
        if(nb == 4):
            for ii in range(len(sk)):
                NM.append(int(sk[ii],base = 16))
        elif(nb == 8):
            NM.append(int(sk,base = 16))
    #Hex 2 dec
    """
    for i in range(len(strl)):
        sk = strl[i]
        NM.append(int(sk,base = 16))"""

    
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
    """
    M16 CTR mdoe of encryption
    Inputs:
        M - plaintext
        Ma - list for mapping
        X - private key matrix
        Y2- private key matrix
        rounds - number of rounds
        t - group order
        delta - private key matrix
        nbits- number of bits for shift function
    Output:
        CF - ciphertext of all rounds
        """
    m = X.shape[0]
    Cnt = np.random.randint(0,np.power(2,t),size=(m,m,1)).astype('int32')
    pp = 5
    Vec = [MakeVector(m) for _ in range(pp)]
    Cnt2 = np.zeros([m,m,1], dtype='int32')
    Cnt2[m-1,m-1,0] = 1
    tt = np.power(2,t)
    CF = np.zeros([m,m,rounds])
    for i in range(rounds):
        if(i % 100 == 0):
            print(i/rounds*100)
        Cnt2 = BuildPlaintext(Vec,m,Cnt2, pp)
        M1 = FormM(BuildPlaintext(Vec,m,np.mod(Cnt+Cnt2,tt),pp), m, Ma, 1)
        Cnt2[m-1,m-1,0] += 1
        pind = m-1
        while(Cnt2[m-1,pind,0] >= tt):
            Cnt2[m-1,pind,0] = np.mod(Cnt2[m-1,pind,0],tt)
            Cnt2[m-1,pind-1,0] += 1
            pind -= 1
        C = M16_Enc(M1[:,:,0,:],delta, X,Y2,t, nbits)
        CF[:,:,i] = np.mod(C+M[:,:,i],tt)#np.mod(C+M[:,:,i],np.power(2,t))#np.mod(np.bitwise_xor(C,M[:,:,i]),np.power(2,t))
    return CF

def M16_CBC(IV,M, Ma, X, Y2, rounds, t, nbits, delta):
    """
    M16 CBC mode of encryption
    Inputs:
        IV - vector of inicialization
        M - plaintext
        Ma - list for mapping
        X - private key matrix
        Y2- private key matrix
        rounds - number of rounds
        t - group order
        delta - private key matrix
        nbits- number of bits for shift function
    Output:
        CF - ciphertext of all rounds
        """
    m = X.shape[0]
    CF = np.zeros([m,m,rounds+1], dtype='int32')
    CF[:,:,0] = IV
    for i in range(rounds):
        M1 = np.mod(CF[:,:,i]+M[:,:,i],np.power(2,t))#np.mod(np.bitwise_xor(CF[:,:,i],M[:,:,i]),np.power(2,t))
        M2 = FormM(M1.reshape([m,m,1]), m, Ma, 1)
        C = M16_Enc(M2[:,:,0,:],delta, X,Y2,t, nbits)
        CF[:,:,i+1] = C
    return CF
