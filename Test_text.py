from modes import Form_text_blocks
import numpy as np
import pickle
import matplotlib.pyplot as plt

m = 4

M1 = Form_text_blocks(m, 'Generated.txt')


file = open(f'res/CTR/AES/Full_text_C','rb')#Generated4_4x4_NoY_addmod_cntr7', 'rb')
C = pickle.load(file)
#C,M = pickle.load(file)

unicode_file = open('Generated.txt',encoding='utf-8')
txt = unicode_file.read()

txt2 = ''
nb = 4
print(C.shape)

unique, counts = np.unique(np.array(C[:,:,:,:]), return_counts=True)
fig, ax = plt.subplots(figsize =(10, 7))
ax.bar(unique, counts,width=1,  edgecolor='black')
print(max(counts))
print(max(counts)//10*10//5)
#plt.yticks(np.arange(0, max(counts), step=max(counts)//10*10//5))
plt.xlabel('Simbolio indeksas')
plt.ylabel('Dažnis')
plt.show()
"""
if(nb == 4):
    C = np.mod(C,16)
    for j in range(M1.shape[2]):
        Mm = C[:,:,j].flatten().astype('int32')
        for i in range(0,16,2):
            v1, v2 = '{:01X}'.format(Mm[i]),'{:01X}'.format(Mm[i+1])
            print(int(v1+v2[-1], base = 16))
            #print(chr(int(v1+v2[-1], base = 16)))
            txt2 += chr(int(v1+v2[-1], base = 16))
elif(nb == 8):
    C = np.mod(C,256)
    for j in range(M1.shape[2]):
        Mm = C[:,:,j].flatten().astype('int32')
        for i in range(0,16):
            v1 = '{:01X}'.format(Mm[i])
            txt2 += chr(int(v1, base = 16))

tst = []
for c in txt:
        tst.append(ord(c))
#text_file = open("res/CTR/M16/Gen_res4.txt", "w",encoding='ascii')
#n = text_file.write(txt2)
#text_file.close()
print(txt2)
"""

"""unique, counts = np.unique(np.array(C[:,:,:]), return_counts=True)
fig, ax = plt.subplots(figsize =(10, 7))
ax.bar(unique, counts,width=1,  edgecolor='black')
#plt.yticks(np.arange(0, max(counts), step=max(counts)//10))
plt.xlabel('Simbolio indeksas')
plt.ylabel('Dažnis')
plt.show()"""


