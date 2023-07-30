from CBC import CBC_F
from CTR import CTR_F
from M16CBC import M16CBC_F
from M16CTR import M16CTR_F
import pickle
from datetime import timedelta

TM = 0
TM2 = 0
TM3 = 0
TM4 = 0
FT = []
FT2 = []
FT3 = []
FT4 = []

nr = 10
for m in range(3,11):
    TM = 0
    TM2 = 0
    TM3 = 0
    TM4 = 0
    for i in range(nr):
        print(f"m = {m}")
        TM += CBC_F(m,'cameraman.tif', nb = 16)
        TM2 += CTR_F(m,'cameraman.tif', nb = 16)
        TM3 += M16CBC_F(m,'cameraman.tif', nb = 16)
        TM4 += M16CTR_F(m,'cameraman.tif', nb = 16)
    FT.append(TM/nr)
    FT2.append(TM2/nr)
    FT3.append(TM3/nr)
    FT4.append(TM4/nr)


file = open(f'res/time_comp_Cameraman_16b', 'wb')
pickle.dump([FT, FT2, FT3, FT4], file)
file.close()
for i in range(len(FT)):
    print(f"m = {i+3}, ",timedelta(seconds=FT[i]))
