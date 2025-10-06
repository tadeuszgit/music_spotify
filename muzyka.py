from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
impo = data_extraction()

raw, wynik = impo.Open_Multiple()
#raw = np.array(raw)
#wynik = np.array(wynik)
#pred = Corr.Coefficient_for_all_dane(dany=raw[-2:], wyniks=wynik[-2:], unsafe=False)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=True)
#Corr.Show_theThing(pred)
#Corr.Show_theThing(np.round(pred*100))
#Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
#print(pred.shape)
s = SpotifyAPI()
order = Corr.ORDER(raw[:], wynik[:], raw[-2:], 0.5, number_songs=100)
print(len(order[-1]))
s.create_new_playlists(order, ['Sezon 3 chapter 3', '25 Sezon 9 chapter 1'])
