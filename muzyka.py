from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
impo = data_extraction()

raw, wynik = impo.Open_Multiple()

#predx = np.vstack(Corr.Prediction_ofBIGCoefficient(raw, wynik))
#print(np.std(wynik[-1], axis=0))
#wprint(np.std(predx[:, -5:], axis=0), np.sum((predx[-80:, -5:] - wynik[-1])**2))
#for i in range(1,100):
    #pred = np.vstack(Corr.Prediction_ofNorm(raw, wynik, save=i))
    #print(i, np.sum(np.where(np.std(pred[:, :], axis=0) < np.std(predx[:, :], axis=0), 1, np.std(predx[:, :], axis=0)/np.std(pred[:, :], axis=0))), np.sum((pred[-82:, -5:] - wynik[-1])**2))
#pred1 = [np.vstack(Corr.Prediction_ofNorm(raw, wynik, save=i)) for i in range(1,100)]
#[print(np.std(pred[:, -5:], axis=0), np.std(pred, axis=0).sum()) for pred in pred1]

#Corr.Show_theThing(pred1[:,-5:])
#print()
#Corr.Show_theThing(pred2[-10:,-5:])
#print()
#Corr.Show_theThing(np.array(wynik[-1])[-10:,-5:])
#print()
#print(np.sum((pred1[:, -5:] - np.array(wynik[-1]))**2),np.sum((pred2[:, -5:] - np.array(wynik[-1]))**2))
Corr.Correlation_betweenSession(raw, wynik)
Corr.lowerdimension(dany=raw, wyniks=wynik)
#input()
#raw = np.array(raw)
#wynik = np.array(wynik)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=False, norm=True)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=True)
#Corr.Show_theThing(pred)
#Corr.Show_theThing(np.round(pred*100))
#Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
#print(pred.shape)
s = SpotifyAPI()
#[Corr.ORDER(raw, wynik, raw, si/100, number_songs=100) for si in range(1,1100)]
order = Corr.ORDER(raw[:], wynik[:], raw[-5:], 0.1, number_songs=100)
print(len(order[-1]))
name = ['Sezon 3 chapter 1','Sezon 3 chapter 2', 'Sezon 3 chapter 3', '25 Sezon 9 chapter 1', '25 Sezon 10 chapter 1']
s.create_new_playlists(order, name[-5:])
