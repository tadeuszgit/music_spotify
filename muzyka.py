from correlation import Correle as Corr
from data_import import data_extraction
import numpy as np
impo = data_extraction()

print("da")
raw, wynik = impo.Open_Multiple()
#raw = np.array(raw)
#wynik = np.array(wynik)
print('na')
pred = Corr.Coefficient_for_all_dane(dany=raw[-2:], wyniks=wynik[-2:], unsafe=False)
pred = (pred - pred.mean(axis=0))/pred.std(axis=0)
Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
print(pred.shape)
Corr.DENSITY(raw[:], wynik[:], 0.5)
