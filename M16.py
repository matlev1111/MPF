import numpy as np
import math
from Basic_functions.main_func import Shifting_bits

def m16mult(a, b, t):
    """
    Multiplication operation defined over M_{2^t}
    Inputs:
        a - First element list consisting of generator a and b powers
        b - Second element list consisting of generator a and b powers
        t - group order
    Output:
        C - list of generator a and b powers
    """
    C = np.zeros([2])
    C[0] = np.mod(a[0]+b[0],2)
    if(np.mod(a[1],2)==0):
        C[1] = np.mod(a[1]+b[1], np.power(2,t-1))
    if(np.mod(a[1],2)==1):
        if(np.mod(b[0],2)==0):
            C[1] = np.mod(a[1]+b[1], np.power(2,t-1))
        else:
            C[1] = np.mod(a[1]+b[1]+np.power(2,t-2),np.power(2,t-1))
    return C


def m16exp(a, n, t):
    """
    Power operation defined over M_{2^t}
    Inputs:
        a - First element list consisting of generator a and b powers
        n - degree of power
        t - group order
    Output:
        b - list of generator a and b powers
    """
    b = np.zeros([2])
    if(np.mod(a[0],2)):
        b[0] = np.mod(n,2)
        if(np.mod(a[1],2)):
            b[1]=np.mod(a[1]*n+np.power(2,t-2)*math.floor(n/2),np.power(2,t-1))
        else:
            b[1]=np.mod(a[1]*n,np.power(2,t-1))
    else:
        b[0]=0
        b[1] = np.mod(a[1]*n,np.power(2,t-1))
    return b


def m16_matrix_exp_right(W,X,t):
    """
    MPF from the right side over M_{2^t}
    Inputs:
        W - base matrix
        X - power matrix
        t - group order
    Output:
        B - result of matrix power function from the right
    """
    m, _, _ = W.shape
    _, n = X.shape
    B = np.zeros([m,n,2])
    for i in range(m):
        for j in range(n):
            B[i,j,0] = 0
            B[i,j,1] = 0
            for k in range(m):
                B[i,j,:] = m16mult(B[i,j,:], m16exp(W[i,k,:], X[k,j],t), t)
    return B.astype('int64')

def m16_matrix_exp_left(W,X,t):
    """
    MPF from the left side over M_{2^t}
    Inputs:
        W - base matrix
        X - power matrix
        t - group order
    Output:
        B - result of matrix power function from the left
    """
    m, _, _ = W.shape
    _, n = X.shape
    B = np.zeros([m,n,2])
    for i in range(m):
        for j in range(n):
            B[i,j,0] = 0
            B[i,j,1] = 0
            for k in range(m):
                B[i,j,:] = m16mult(B[i,j,:], m16exp(W[k,j,:], X[i,k],t), t)
    return B.astype('int64')


def M16_Enc(M,delta, X,Y,t, nbits):
    """
    Encryption algorithm defined over M_{2^t}
    Inputs:
        M - plaintext
        delta - private key
        X - private key
        Y - private key
        t - group order
        nbits - number of bits for the shift
    Output:
        C4 - Ciphertext
    """
    m,_ = X.shape
    M_inp = M.copy()
    M_inp[:,:,1] = np.mod(M_inp[:,:,1]+X,np.power(2,t-1))
    M_inp[:,:,0] = np.mod(delta[:,:,0]+M_inp[:,:,0],2)
    C1 = M_inp
    C2 = m16_matrix_exp_left(C1,Y,t)
    C2_2 = m16_matrix_exp_right(C2,Y,t)
    C3 = np.zeros([m,m])
    DX = C3.copy()
    for i in range(m):
        for j in range(m):
            C3[i,j] = int(Shifting_bits(bin(C2_2[i,j,0])+bin(C2_2[i,j,1])[2:].zfill(t-1),nbits),2)
            DX[i,j] = np.mod(int(bin(delta[i,j,0])+bin(X[i,j])[2:].zfill(t-1),2),np.power(2,t))
    C4 = np.mod(C3+DX, np.power(2,t))
    return C4.astype('int64')


def M16_Dec(C,delta, X,YY,t, nbits):
    """"
    Decryption algorythm defined over M_{2^t}
    Inputs:
        C - ciphertext
        delta - private key
        X - private key
        YY - inverse of private key Y
        t - group order
        nbits - number of bits for a shift
    Outputs:
        D5 - deciphered ciphertext
    """
    m,_ = X.shape
    DX = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            DX[i,j] = np.mod(int(bin(delta[i,j,0])+bin(X[i,j])[2:].zfill(t-1),2),np.power(2,t))
    D1 = np.mod(C-DX, np.power(2,t)).astype('int64')
    D2 = np.zeros([m,m,2])
    D5 = D2.copy()
    for i in range(m):
        for j in range(m):
            PD = Shifting_bits('0b'+bin(D1[i,j])[2:].zfill(t),t-nbits)
            D2[i,j,0] = int(PD[:2]+PD[2],2)
            D2[i,j,1] = int(PD[:2]+PD[3:],2)
    D3 = m16_matrix_exp_right(D2, YY,t)
    D4 = m16_matrix_exp_left(D3, YY,t)
    D5[:,:,1] = np.mod(D4[:,:,1]-X, np.power(2,t-1))
    D5[:,:,0] = np.mod(delta[:,:,0]+D4[:,:,0],2)
    return D5.astype('int64')


def FormM(C, m, Ma, amount):
    """
    Mapping M elements to M_b & M_a elements
    Inputs:
        C - matrix which needs to be mapped
        m - matrix order
        Ma - List for mapping
        amount - number of matrices for mapping
    Output:
        M - mapped matrix
    """
    M = np.zeros([m,m,amount,2])
    for a in range(amount):
        #print(a/amount*100)
        for i in range(m):
            for j in range(m):
                M[i,j,a,0] = Ma[C[i,j, a],0]
                M[i,j,a,1] = Ma[C[i,j, a],1] 
    return M.astype('int64')


def perm_matrix(n,t):
    """
    Permutation matrix generation
    Inputs:
        n - matrix order
        t - group order
    Output
        D - permutation matrix
    """
    A = np.eye(n)
    A = np.random.permutation(A)
    B = np.mod(np.random.randint(1,np.power(2,t-2)+1,size=(n,n))*2,np.power(2,t-1))
    pag = np.mod(np.random.randint(1,np.power(2,t-2)+1,size=(n,n))*2-1,np.power(2,t-1))
    D = np.mod(B-B*A+pag*A, np.power(2,t-1))
    return D.astype('int64')

def MakeVector(m):
    """
    Constructin permutation vector
    Inputs:
        m - matrix order
    Output:
        Permutation vector
    """
    Vec = np.arange(0,m*m)
    return np.random.permutation(Vec)

def BuildPlaintext(VVec, m, M,pp):
    """
    Form plaintext using permutation vector mapping
    Inputs:
        VVec - Permutation vectors
        m - matrix order
        M - Plaintext
        pp - group order for permutation vectors
    Output:
        NM - Mapped maprix M
    """
    if(VVec == None):
        return M
    index = 0
    Vec = VVec[np.mod(M.sum(dtype='int64'),pp)]
    NM = np.zeros(shape=M.shape)
    for i in range(m):
        for j in range(m):
            NM[i,j] = M[Vec[index]//m,Vec[index]%m]
            index += 1
    return NM.astype('int64')
