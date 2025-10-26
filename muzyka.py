from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
impo = data_extraction()

raw, wynik = impo.Open_Multiple()
analyse = impo.Open_Multiple(both=False, all_com=False)[:-1]
##LAMBDA SEARCH

"""for i in range(1):
    LAMBDA = 4.2955*(4.2959/4.2955)**(i/100)
    a = 93.31500
    b = 93.31510
    LAMBDA = a * (b/a) ** (i/100) 
    LAMBDA = 0
    pred = raw[:]
    LAMBDA = [5,1,8,20,22,21,16,11]
    LAMBDA = [0,0.08,11,12,10,9,8,8,7,7,7,7,6,6,6,6,5,5,5,5,5]
    for L in LAMBDA:
        coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=True, LAMBDA=L)
        coef_acc = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=L)
        acc = [Corr.Prediction_ofCoefficient(coeffiecient=coef_acc, dane_onlyX=pre, norm=False) for pre in pred]
        error = [np.mean(np.mean((acc[j][:, j*5:j*5+5] - wynik[j])**2, axis=0)**0.5/np.std(wynik[j],axis=0)) for j in range(len(acc))]
        error = np.mean(error)
        pred = [Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pre, norm=False) for pre in pred]
        #print(np.vstack(pred).shape)
        #print(L, np.std(np.std(np.vstack(pred), axis=0)), np.median(np.std(np.vstack(pred), axis=0)), error)
        print(acc[-1][-1:,-5:])
        pred = [1 / (1 + np.exp(-pre)) for pre in pred]
    #coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=6)
    #pred = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pred[-1], norm=False)
    #print(len(pred))
    #input()
    #Corr.Show_theThing(pred[:, -5:])
    #print(np.mean(np.mean((pred[:, -5:] - wynik[-1])**2, axis=0)**0.5/np.std(wynik[-1],axis=0)))
""""""lolly=[]
test = -18
for k in range(1000):
    ridx = np.arange(len(raw[test]))
    np.random.shuffle(ridx)
    total_error = []
    #print(ridx)
    for i in range(100):
        j = 1.1 ** (i+0)
        coef = Corr.Coefficient(raw[test][ridx[:50]], wynik[test][ridx[:50]], LAMBDA=j)
        pred = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=raw[test][ridx[50:]])
        #print(pred.shape)
        error = ((pred - wynik[test][ridx[50:]])/np.std(wynik[test])) ** 2
        #Corr.Show_theThing(pred)
        #print(j, error.mean())
        total_error.append(error.mean())
        varie = pred.std(axis=0)
        information = coef.T ** 2 @ np.hstack((0, np.std(raw[-1],axis=0)))[:, None]
        score = varie[:, None] ** 5 / (information ** 2)
        #print(j, np.sum(varie ** 2), np.sum(information), score.sum())
    #print()
    lolly.append(1.1**(np.argmin(total_error)+0))
    print(k, 1.1** (np.argmin(total_error)+0), min(total_error), np.median(lolly))
    
print()
print(np.median(lolly), np.mean(lolly))"""
#[print(coe.sum()) for coe in coef]
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
Corr.Correlation_betweenSession(raw, wynik, test=analyse[:-1])
#Corr.lowerdimension(dany=raw, wyniks=wynik)
#input()
#raw = np.array(raw)
#wynik = np.array(wynik)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=False, norm=True)
#pred = Corr.Coefficient_for_all_dane(dany=raw[:], wyniks=wynik[:], unsafe=True)
#Corr.Show_theThing(pred)
#Corr.Show_theThing(np.round(pred*100))
#Corr.Show_theThing(np.round(pred[:, 4::5]*100)/100)
#print(pred.shape)
#s = SpotifyAPI()

#print(lol.shape)
name = ['Sezon 2 chapter 1','Sezon 2 chapter 2','Sezon 2 chapter 3','Sezon 2 chapter 4','Sezon 3 chapter 1','Sezon 3 chapter 2', 'Sezon 3 chapter 3', '25 Sezon 9 chapter 1', '25 Sezon 10 chapter 1', '25 Sezon 10 chapter 2', f"Lumyn's Mixtape"]
name = name[:]
analyse = analyse
lol = Corr.Prediction_naSterydach(raw, wynik, test=analyse[-len(name):], norm=True)
order = Corr.GMM(np.vstack(lol), k_mean=11)
#order = Corr.K_mean(np.vstack(lol))
#name = [f"Lumyn's Mixtape"]
#[Corr.ORDER(raw, wynik, raw[-len(name):], si/100, number_songs=100) for si in range(2,3)]
#order = Corr.ORDER(raw[:], wynik[:], analyse[-len(name):], 4, number_songs=10)
#input()
#order = Corr.ORDER(raw[:], wynik[:], analyse[-len(name):], 40, number_songs=1000, similar=order[0])
Corr.Show_theThing(order)
#print(len(order[-1]))
s.create_new_playlists(order, name, name_new_playlists=[str(i) for i in range(len(order))])
name_new_playlists=['Lym Gym', 'Lym Past', 'Lym Better', 'Lym idk', 'Lym Like']
