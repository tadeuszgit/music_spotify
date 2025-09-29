from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
impo = data_extraction()

print("da")
raw, wynik = impo.Open_Multiple()
#raw = np.array(raw)
#wynik = np.array(wynik)
print('na')
pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=True)
pred = (pred - pred.mean(axis=0))/pred.std(axis=0)
#Corr.Show_theThing(np.round(pred*100))
#Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
#print(pred.shape)
#s = SpotifyAPI()
order = Corr.ORDER(raw[:], wynik[:], raw[-1:], 0.5)
#s.create_new_playlists(order, ['Sezon 3 chapter 3'])
