import pickle
import numpy as np
from PIL import Image

file = open('res/linux', 'rb')
data, s1, s2 = pickle.load(file)
file.close()
p = 563
q = (p-1)/2
datam = np.mod(data,q)
print(data.shape)
"""s1 = 392
s2 = 520"""
def Print_pic(C, s1, s2, d3):
    m = C[:,:,0].shape[0]
    if(d3):
        rgb = 3
        FC = np.zeros([s1,s2,3])
    else:
        rgb = 1
        FC = np.zeros([s1,s2])
    indx = 0
    for k in range(rgb):
        for i in range(s1//m):
            for j in range(s2//m):
                if(d3):
                    FC[i*m:(i+1)*m,j*m:(j+1)*m,k] = C[:,:,indx]
                else:
                    FC[i*m:(i+1)*m,j*m:(j+1)*m] = C[:,:,indx]
                indx += 1
    print(np.array(FC))
    if(d3):
        im = Image.fromarray(FC[:,:,:].astype(np.uint8)).convert('RGB')
    else:
        im = Image.fromarray(FC).convert('L')
    return im

ans = Print_pic(datam, s1, s2, 0)
ans.show()
