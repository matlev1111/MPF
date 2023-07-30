from PIL import Image, ImageDraw
import numpy as np
import math

im = Image.open(f"images/cameraman.tif").convert('RGB')#Image.open("coloredChips.png")
X1 = np.asarray(im)

"""indx = 0
sh = X1.shape
for i in range(sh[0]):
    for j in range(sh[1]):
        if(i%5==0 or j%5==0):
            X1[i,j] = 
print(X1.shape)"""
#X1[:,256] = np.zeros([256,2])
X2 = np.zeros([266,266,3]) + 255

X2[:256,:256,:] = X1


im2 = Image.fromarray(np.uint8(X2))

"""drw = ImageDraw.Draw(im)
for i in range(-1,X1.shape[1],128):
    if i < 0:
        i = 0
    drw.line((i,0)+(i,X1.shape[1]), fill=(255,0,0))
    drw.line((0,i)+(X1.shape[1],i), fill=(255,0,0))
im.save("../pav2.png")"""

drw = ImageDraw.Draw(im2)
for i in range(-1,X2.shape[1],133):
    if i < 0:
        i = 0
    drw.line((i,0)+(i,X2.shape[1]), fill=(255,0,0))
    drw.line((0,i)+(X2.shape[1],i), fill=(255,0,0))
im2.save("../pav1.png")
# 256
# 6 i apacia
"""kk = 256*256//266+1
while (kk % 133 != 0):
    kk += 1

print(kk)"""


"""XX = np.zeros([256*256,3])

X3 = np.zeros([X1.shape[1]//133*133,(256*256//(133*133)+1)*133,3])
print(X1.shape[1]//133)
print(256*256//(133*133)+1)
m2 = 133
flg = 0
for i in range(256*256//(m2*m2)+1):
    for j in range(X1.shape[1]//m2+1):

        X3[i*m2:(i+1)*m2,j*m2:(j+1)*m2] = X1[i*m2:(i+1)*m2,j*m2:(j+1)*m2,:]
        flg = 0
        if(X1[i*m2:(i+1)*m2,(j+1)*m2:,:].shape[1] < m2):
            X3[(i+1)*m2:(i+2)*m2,0:(X1.shape[1]%133)] = X1[i*m2:(i+1)*m2,(j+1)*m2:,:]
            flg = 1



X1[0:133,0:133,3]
nln = X1[133:266,:,:]
nln2 = X1[0:133,133:,3]+nln"""
"""m2 = 120
kk = math.ceil(256*256//(m2*m2))*m2


X3 = np.zeros([m2,7*m2,3])
print(X3.shape)
print(m2, kk)
print(math.ceil(X1.shape[0]/m2),math.ceil(X1.shape[1]/m2))
pind = 0
indx = m2
for i in range(math.ceil(X1.shape[0]/m2)):
    for j in range(math.ceil(X1.shape[1]/m2)): 
        print(pind,indx)
        if(X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[0] < m2):
            tmp=X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[0]
            if(X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[1] < m2):
                indx = pind + X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[1]
                X3[:tmp,pind:indx,:] = X1[i*m2:, (j)*m2:,:]
                pind = indx
                indx = pind + m2
            else:
                print(tmp)
                print(i*m2)
                print(pind, indx)
                print(j*m2,(j+1)*m2)
                X3[:tmp,pind:indx,:] = X1[i*m2:, j*m2:(j+1)*m2,:]
                pind = indx
                indx = pind + m2
        else:
            if(X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[1] < m2):
                indx = pind + X1[i*m2:(i+1)*m2, (j)*m2:,:].shape[1]
                X3[:,pind:indx,:] = X1[i*m2:(i+1)*m2, (j)*m2:,:]
                pind = indx
                indx = pind + m2
            else:
                X3[:,pind:indx,:] = X1[i*m2:(i+1)*m2, j*m2:(j+1)*m2,:]
                pind = indx
                indx = pind + m2"""

"""XX3 = np.zeros([(256*256//(m2*m2)+1)*m2,X1.shape[1]//m2*m2,3])
print(XX3.shape)
indx = 0
for i in range(XX3.shape[0]//m2):
    for j in range(XX3.shape[1]//m2):
        XX3[i*m2:(i+1)*m2,j*m2:(j+1)*m2,:] =X3[:,indx*m2:(indx+1)*m2,:]
        indx += 1 """

"""im3 = Image.fromarray(np.uint8(X3))
im3.show()"""
"""drw = ImageDraw.Draw(im3)
for i in range(266//2):
    drw.line(((i+1)*2,0)+((i+1)*2,X3.shape[0]), fill=(255,0,0))
for i in range(kk//2):
    drw.line((0,(i+1)*2)+(X3.shape[1],(i+1)*2), fill=(255,0,0))"""
#im3.save("../pav3.png")
#65536
#65436