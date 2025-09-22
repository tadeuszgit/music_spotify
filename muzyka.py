from correlation import Correle as Corr
from data_import import data_extraction
import numpy as np
impo = data_extraction()

print("da")
raw, wynik = impo.Open_Multiple()
#raw = np.array(raw)
#wynik = np.array(wynik)
print('na')
pred = Corr.Prediction_ofCorrelation(Corr.Correlation(dane=raw[-1], wynik=wynik[-1]), dane_onlyX=raw[-1])
Corr.Show_theThing(pred)
print()
diff = (pred - wynik[-1])
Corr.Show_theThing(diff)
print()
diff = diff ** 2
diff = np.mean(diff, axis=0, keepdims=True) ** 0.5
Corr.Show_theThing(diff)
Corr.Show_theThing(np.std(wynik[-1], axis=0, keepdims=True))