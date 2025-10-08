from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
impo = data_extraction()

raw, wynik = impo.Open_Multiple()
"""pred1 = Corr.Prediction_ofNorm(raw, wynik)[-1]
pred2 = Corr.Prediction_ofBIGCoefficient(raw, wynik)[-1]
Corr.Show_theThing(pred1[-10:,-5:])
print()
Corr.Show_theThing(pred2[-10:,-5:])
print()
Corr.Show_theThing(np.array(wynik[-1])[-10:,-5:])
print()
print(np.sum((pred1[:, -5:] - np.array(wynik[-1]))**2),np.sum((pred2[:, -5:] - np.array(wynik[-1]))**2))"""
#Corr.Correlation_betweenSession(raw, wynik)
Corr.lowerdimension(dany=raw, wyniks=wynik)
#raw = np.array(raw)
#wynik = np.array(wynik)
#pred = Corr.Coefficient_for_all_dane(dany=raw[-2:], wyniks=wynik[-2:], unsafe=False)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=True)
#Corr.Show_theThing(pred)
#Corr.Show_theThing(np.round(pred*100))
#Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
#print(pred.shape)
#s = SpotifyAPI()
order = Corr.ORDER(raw[:], wynik[:], raw[-4:], 0.5, number_songs=100)
print(len(order[-1]))
#s.create_new_playlists(order, ['Sezon 3 chapter 1','Sezon 3 chapter 2', 'Sezon 3 chapter 3', '25 Sezon 9 chapter 1'])
