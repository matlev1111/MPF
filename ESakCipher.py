import numpy as np
from Basic_functions.main_func import *
from Basic_functions.Gauss_Jordan import Gauss_Jordan

def GenG(p):
    """
    Generates index and number list of the mapping function
    """
    q = (p-1)//2
    index = 3
    while (mod_exp(index,q,p) != 1 or np.mod(mod_exp(index,1,p) * mod_exp(index,2,p),q) == mod_exp(index,2,p)):
        index = index + 1
    numbers = []
    for i in range(1,q+1):
        numbers.append(mod_exp(index,i,p))
    return numbers


def Gmap(index, mat):
    """
    Maps matrix elements using function f
    """
    try:
        m = len(mat)
    except:
        m = 1
    M = np.zeros(m)
    fm = np.vectorize(lambda x: index[x])
    M=fm(mat)
    return M.astype('int64')

def imap(ind, num):
    """
    Performs an inverse mapping F^{-1}
    """
    try:
        m = len(num)
    except:
        m = 1
        return ind.index(num)
    M = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            found = ind.index(num[i,j])
            M[i,j] = found
    return M.astype('int64')

def ESakCipher(G,X,M,Y,p,q,Fmap):
    """
    G = GenG(p)
    X \in 
    M \in 
    Y \in Z_q \ {0}
    """
    #q = (p-1)//2
    #Fmap = Gmap(G,X)
    C1 = np.mod(X+M,q).astype('int64')
    F = Gmap(G,C1)
    YF = matrix_exp_left(F,Y,p)
    YFY = matrix_exp_right(YF,Y,p)
    C2 = hadamard_prod(Fmap, YFY,p)
    return np.mod(imap(G, C2)+X,q).astype('int64')

def ESakDecryption(G, YY, X, C,p):
    """
    Decryption algorythm
    """
    q = (p-1)//2
    FF = hadamard_inv(Gmap(G,X),p)
    prod = hadamard_prod(FF,C,p)
    YYprod = matrix_exp_right(prod,YY,p)
    YYprodYY = matrix_exp_left(YYprod,YY,p)
    M_new = np.mod(imap(G,YYprodYY,p)-X,q)
    return M_new

def Gen_parameters(m, p):
    """
    Parameters generation
    """
    q = (p-1)/2
    G = GenG(p)
    X = np.random.randint(0,q,size=(m,m))
    Y = np.random.randint(1,q,size=(m,m))
    Z = np.random.randint(0,q,size=(m,m))
    YY = Gauss_Jordan(Y,q)
    while(np.mod(np.linalg.det(Y),q)==0 or (np.mod(np.matmul(Y,YY),q) == np.eye(m)).sum() != m*m):
        Y = np.random.randint(1,q,size=(m,m))
        YY = Gauss_Jordan(Y,q)
    return q, G, X, Y, Z
