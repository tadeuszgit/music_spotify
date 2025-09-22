from correlation import Correle as Corr
from data_import import data_extraction
import numpy as np
impo = data_extraction()

print("da")
raw, wynik = impo.Open_Multiple()
#raw = np.array(raw)
#wynik = np.array(wynik)
print('na')
pred = Corr.Correlation_for_all_dane(dany=raw[-2:], wyniks=wynik[-2:], unsafe=False)
pred = (pred - pred.mean(axis=0))/pred.std(axis=0)
Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
print(pred.shape)
pred = [Corr.Prediction_ofCorrelation(Corr.Correlation(dane=raw[i], wynik=wynik[i]), dane_onlyX=raw[i]) for i in range(len(raw))]
#Corr.Show_theThing(np.vstack(pred))
print("fd")
input()
diff = [(pred[i] - wynik[i]) ** 2 for i in range(len(pred))]
Corr.Show_theThing(np.vstack(diff))
diff = np.array([np.mean(dif, axis=0) ** 0.5 for dif in diff])
print()
Corr.Show_theThing(diff)
stdd = np.array([np.std(wyn, axis=0) for wyn in wynik])
print()
Corr.Show_theThing(stdd)
print()
Corr.Show_theThing(stdd - diff)
print()
pred = Corr.Prediction_ofCorrelation(Corr.Correlation(dane=raw[-1], wynik=wynik[-1]), dane_onlyX=raw[-1])
diff = (pred - wynik[-1]) ** 2
diff = np.mean(diff, axis=0, keepdims=True) ** 0.5
Corr.Show_theThing(diff)
Corr.Show_theThing(np.std(wynik[-1], axis=0, keepdims=True))