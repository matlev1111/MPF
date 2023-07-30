import numpy as np

def Gauss_Jordan(Am,p):
    """
    Find an inverse of a matrix by applying Gauss-Jordan method
    Input:
        Am - Matrix which inverse we want to find
        p - prime number of a groupd matrix is defined of
    Output:
        Inverse matrix, or empty list if inverse matrix is non existing
    """
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
    """
    Second step of Gauss-Jordan method making top right zeroes triange
    Input:
        A - Initial matrix
        B - matrix helper, later inverse matrix
        a - index of diagonal elements
        n - matrix order
        p - group order
    Output:
        A - modified matrix, at the end identitiy matrix
        B - modified matrix, at the end inverse matrix
    """
    for i in reversed(range(a)):
        mult = p-np.mod(A[i,a],p)
        for j in range(n):
            A[i,j] = np.mod(A[i,j]+mult*A[a,j],p)
            B[i,j] = np.mod(B[i,j]+mult*B[a,j],p)
    return A,B

def Change(A,n,a,pab,p):
    """
    Add bottom row to the top if element [a,a] is zero
    Input:
        A - Initial matrix
        n - matrix order
        a - index of [a,a] element
        pab - non-zero element row index
        p - hroup order
    Output:
        A - modified matrix
    """
    for j in range(n):
        A[a,j] = np.mod(A[a,j]+A[pab,j],p)
    return A

def Search(n, A,a):
    """
    Searching for non-zero element in a column and returning index
    Input:
        n - matrix order
        A - Initial matrix
        a - index of a column
    Output:
        index of a row with non-zero element
    """
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
    """
    Dividing row elements by a specific number (multiplied by an inverse)
    Input:
        A - Initial matrix which  we want to change
        el - element which inverse we are looking for
        n - matrix order
        p - order of a group
        i - row index
        tikr - boolean allowing to tell if we can find inverse matrix or not
        B - identity matrix at first (changing after each step)
    """
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
    """
    Changing sign of a row
    Input:
        A - Initial matrix which signs we want to change
        n - matrix order
        p - order of a group
        i - row index
    Output:
        A - Modified initial matrix
    """
    for j in range(n):
        A[i,j] = A[i,j]+p
    return A

def In(A,B,a,n,p,tikr):
    """
    Gauss-Jordan going down step, making bottom zeroes triangle
    Input:
        A - Matrix which inverse we want to find
        B - identity matrix at first (changing after each step)
        a - index for main diagonal
        n - matrix order
        p - order of the group
        tikr - boolean allowing to tell if we can find inverse matrix or not
    Output:
        A - modified initial matrix
        B - modified initial matrix
        tikr - boolean allowing to tell if we can find inverse matrix or not
    """
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
            