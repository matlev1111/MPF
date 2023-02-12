import numpy as np

def Gauss_Jordan(Am,p):
    A = Am.copy()
    n = len(A)
    B = np.eye(n). astype('int32')
    tikr = 1
    for i in range(n):
        A,B,tikr = In(A,B,i,n,p,tikr)
        if(not(tikr)):
            B = []
            print('Inverse matrix does not exist!')
            break
    if(tikr):
        if(A[n-1,n-1]<0):
            A = Sign(A,n,p,n-1)
            B = Sign(B,n,p,n-1)
        if(A[n-1,n-1]!= 1):
            if(np.mod(A[n-1,n-1],p)==0):
                tikr = 0
            else:
                A,B,tikr = Division(A, A[n-1,n-1],n,p,n-1,tikr,B)
        if(tikr):
            for i in reversed(range(n)):
                A,B = Out(A,B,i,n,p)
    return B


def Out(A,B,a,n,p):
    for i in reversed(range(a)):
        mult = p-np.mod(A[i,a],p)
        for j in range(n):
            A[i,j] = np.mod(A[i,j]+mult*A[a,j],p)
            B[i,j] = np.mod(B[i,j]+mult*B[a,j],p)
    return A,B

def Change(A,n,a,pab,p):
    for j in range(n):
        A[a,j] = np.mod(A[a,j]+A[pab,j],p)
    return A

def Search(n, A,a):
    for i in range(a,n):
        if(np.mod(A[i,a],2)!= 0):
            return i
    return -1

def extended_euclid_gcd(a, b):
    """
    Returns a list `result` of size 3 where:
    Referring to the equation ax + by = gcd(a, b)
        result[0] is gcd(a, b)
        result[1] is x
        result[2] is y 
    """
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a
    while r != 0:
        quotient = old_r//r 
        old_r, r = r, old_r - quotient*r
        old_s, s = s, old_s - quotient*s
        old_t, t = t, old_t - quotient*t
    return [old_r, old_s, old_t]

def Division(A,el,n,p,i,tikr,B):
    d,x,_ = extended_euclid_gcd(el, p)
    if(d and tikr):
        sk = np.mod(x,p)
        for j in range(n):
            A[i,j] = np.mod(A[i,j]*sk,p)
            B[i,j] = np.mod(B[i,j]*sk,p)
    else:
        tikr = 0
    return A, B, tikr

def Sign(A,n,p,i):
    for j in range(n):
        A[i,j] = A[i,j]+p
    return A

def In(A,B,a,n,p,tikr):
    if(tikr):
        if(A[a,a]<0):
            A = Sign(A,n,p,a)
            B = Sign(B,n,p,a)
        if(tikr):
            if(np.mod(A[a,a],2)==0):
                s = Search(n,A,a)
                if(s < 0):
                    A,B, tikr = Division(A, A[a,a], n, p, a, tikr, B)
                    return A,B, tikr
                if(p != -1):
                    A = Change(A, n, a, s, p)
                    B = Change(B, n, a, s, p)
                    if(A[a,a] < 0):
                        A = Sign(A, n, s, a)
                        B = Sign(B, n, s, a)
                    if(A[a,a] != 0 and A[a,a] != 1):
                        A,B,tikr = Division(A, A[a,a],n,p,a,tikr,B)
                else:
                    tikr = 0
            if(A[a,a] != 0 and A[a,a] != 1):
                A, B, tikr = Division(A, A[a,a], n, p, a, tikr, B)
            for i in range(a+1,n):
                mult = p-np.mod(A[i,a],p)
                for j in range(n):
                    A[i,j] = np.mod(A[i,j]+mult*A[a,j],p)
                    B[i,j] = np.mod(B[i,j]+mult*B[a,j],p)
    return A,B, tikr
            