import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

plt.rcParams['text.usetex'] = True
file = open(f'res/time_comp_Cameraman_16b', 'rb')
L1CBC, L2CTR, L3M16CBC, L4M16CTR = pickle.load(file)
xlab = ['CBC', 'CTR', '']
print(L1CBC)
print(L2CTR)
print(L3M16CBC)
print(L4M16CTR)
lbl = []
for i in range(len(L3M16CBC)):
    if(len(L1CBC) > 0 and len(L3M16CBC) > 0):
        lbl.append(f'CBC (m={3+i})')
        lbl.append(f'CTR (m={3+i})')
        lbl.append(r'$M_{2^t}CBC$ '+f'(m={3+i})')
        lbl.append(r'$M_{2^t}CTR$ '+f'(m={3+i})')
        plt.bar(i+0, L1CBC[i], color='#fca289', width = 0.2)
        plt.bar(i+0.25, L2CTR[i], color = '#eec448', width = 0.2)
        plt.bar(i+0.5, L3M16CBC[i], color='#aacc81', width = 0.2)
        plt.bar(i+0.75, L4M16CTR[i],color='#86c5bf', width = 0.2)
    elif (len(L3M16CBC) > 0):
        lbl.append(r'$M_{2^t}CBC$ '+f'(m={3+i})')
        lbl.append(r'$M_{2^t}CTR$ '+f'(m={3+i})')
        plt.bar(i, L3M16CBC[i], color='#aacc81', width = 0.2)
        plt.bar(i+0.5, L4M16CTR[i],color='#86c5bf', width = 0.2)
    
if(len(L1CBC) > 0 and len(L3M16CBC) > 0):
    plt.xticks([0.25*i for i in range(len(lbl))], lbl, size='small')
elif (len(L3M16CBC) > 0):  
    plt.xticks([0.5*i for i in range(len(lbl))], lbl, size='small')
plt.xticks(rotation=85)
plt.ylabel('Laikas (s)')
plt.tight_layout()
plt.show()