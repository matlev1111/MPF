import numpy as np
from sympy.ntheory import discrete_log
from main_func import *

def GenG(p):
    """
    Generates index and number list of the mapping function
    """
    q = (p-1)//2
    index = 3
    while (mod_exp(index,q,p) != 1 and mod_exp(index,1,p) * mod_exp(index,2,p) != mod_exp(index,1,p)):
        index = index + 1
    numbers = []
    for i in range(q):
        numbers.append(mod_exp(index,i,p))
    return index, numbers


def Gmap(index, mat):
    """
    Maps matrix elements using function f
    """
    m = len(mat)
    M = np.zeros(m)
    fm = np.vectorize(lambda x: index[x])
    M=fm(mat)
    return M.astype('int32')

def imap(ind, num,p):
    """
    Obtains pre-image of each matrix element
    """
    m = len(num)
    M = np.zeros(m)
    fm = np.vectorize(lambda x: discrete_log(p, x, ind))
    M=fm(num)
    return M

def ESakCipher(G,X,M,Y,p,q,Fmap):
    """
    G = GenG(p)
    X \in 
    M \in 
    Y \in Z_q \ {0}
    """
    #q = (p-1)//2
    #Fmap = Gmap(G,X)
    C1 = np.mod(X+M,q).astype('int32')
    F = Gmap(G,C1)
    YF = matrix_exp_left(F,Y,p)
    YFY = matrix_exp_right(YF,Y,p)
    C2 = hadamard_prod(Fmap, YFY,p)
    return C2

def ESakDecryption(G,ind, YY, X, C,p):
    q = (p-1)//2
    FF = hadamard_inv(Gmap(G,X),p)
    prod = hadamard_prod(FF,C,p)
    YYprod = matrix_exp_right(prod,YY,p)
    YYprodYY = matrix_exp_left(YYprod,YY,p)
    M_new = np.mod(imap(ind,YYprodYY,p)-X,q)
    return M_new


