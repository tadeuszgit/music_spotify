from correlation import Correle as Corr
from data_import import data_extraction
from spotifyAPI import SpotifyAPI
import numpy as np
from corr_fl import Corrl
impo = data_extraction()
raw, wynik = impo.Open_Multiple()
analyse = impo.Open_Multiple(both=False, all_com=False)[:-1]
test = Corrl()
m = test.Coefficient_for_all_dataset(LAMBDA=0)
print("DONE!!!!!")
print(m.shape)
input()
def season(name_play, pos, raw, wynik, analyse, s, name):
    order = []
    matrix = Corr.Prediction_naSterydach(raw, wynik, test=analyse, norm=True)
    for i in range(len(pos)-1):
        oder = Corr.ORDER(SIGMA=4, matrix=matrix, number_songs=500, periods=pos[i+1]-pos[i], uniq=pos[i]*5+4+15, start=pos[i])
        oder = np.array(oder)
        oder += sum([len(an) for an in analyse[:pos[i]]])
        order.append(oder[-1])
        print(oder)
        print(sum([len(an) for an in analyse[:pos[i]]]))
        print(pos[i+1]-pos[i])
        print(pos[i]*5+4+15)
        input()
    Corr.Show_theThing(np.array(order)[:, :])
    input()
    s.create_new_playlists(order, name, name_new_playlists=name_play)
#coef = Corr.Coefficient_for_all_dane(dany=raw, wyniks=wynik, LAMBDA=0, norm=True, unsafe=True)
#Corr.Coefficient_APPROX(dane=Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=raw[-1]), wynik=wynik[-1], LAMBDA=1)
#s
##LAMBDA SEARCH
"""pred = raw[:]
coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=True, LAMBDA=0)
pred = [Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pre, norm=True) for pre in pred]
coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=True, LAMBDA=0.11)
pred = [Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pre, norm=True) for pre in pred]
#mask = [4, 11, 21, 44, 67, 83, 108, 111, 114]
#mask = [4, 5, 6, 11, 13, 16, 26, 27, 28, 33, 34, 36, 41, 42, 46, 48, 51, 53, 57, 62, 63, 67, 70, 71, 73, 75, 76, 81, 88, 95, 96, 97,]
#mask = [4, 5, 6, 16, 26, 27, 28, 36, 46, 51, 53, 62, 67, 70, 71, 73, 75, 76,88, 95, 96, 97,102,106,107,113]
#mask = [5, 16, 27, 28, 46, 67, 76, 96,102,106,107,113]
pred = [pre[:, :] for pre in pred]
feature_std = np.hstack((0, np.std(np.vstack(pred), axis=0)))
goal_std = np.hstack([np.std(wyn, axis=0) for wyn in wynik])
tren = 0
trend = 0
dim = 1
Cel = 70
lista = np.arange(len(goal_std))
print(lista)
for k in range(500):
    for j in range(20):
        for i in range(200):
            coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=tren)
            TRUE_ENERGY = np.sum((coef[:, 4:5] ** 2) ** (dim / 2) * feature_std[:, None] ** dim / goal_std[None, 4:5] ** dim)
            #Cel = 5700
            MX = TRUE_ENERGY / Cel
            error = Corr.Prediction_ofCoefficient(coeffiecient=coef[:, 4:5], dane_onlyX=pred[0], norm=False)
            error -= wynik[0][:, 4:5]
            error = np.mean(error ** 2) ** 0.5
            punish = 2 * np.log(MX)/MX * dim / (goal_std[4] ** dim * Cel)
            lahm = punish * feature_std[:, None] ** dim * ((coef[:, 4:5] ** 2) ** ((dim-2)/2))
            lahm = np.where(np.abs(lahm) < 10 ** 100, lahm, 10 ** 100 * np.sign(lahm))
            #Corr.Show_theThing(np.eye(len(lahm)) * lahm)
            lahm = np.eye(len(lahm)) * lahm
            tren = lahm * 0.1 + tren * 0.9
            #print(trend, TRUE_ENERGY, punish, error)
            trend = float(punish * 0.1 + trend * 0.9)
        Cel = TRUE_ENERGY
        print("CEL", Cel , coef[:, 4:5].shape, np.where(np.abs(coef[:, 4:5]) > 0.01)[0].shape, error)
        trend = 0
        tren = 0
        #print(coef[:, 4:5].shape)
        #print(np.where(np.abs(coef[:, 4:5]) > 0.01)[0].shape)
        #input()
    mask = np.argsort(np.abs(coef[1:, 4:5]), axis=0)[:, 0]
    min_tres = 0
    while np.abs(coef[mask[min_tres+1]+1, 4]) < 0.01:
        min_tres += 1
    if np.abs(coef[mask[min_tres]+1, 4]) > 0.01:
        break
    print(mask[min_tres], coef[mask[min_tres]+1, 4])
    print("REMOVED", lista[mask[min_tres]])
    mask = np.sort(mask[min_tres+1:])
    #print(mask)
    feature_std = feature_std[np.hstack((0, mask+1))]
    pred = [pre[:, mask] for pre in pred]
    lista = lista[mask]
    print(lista)
    tren = 0
    trend = 0
    #Corr.Show_theThing(coef[:, 4:5])
    #Corr.Show_theThing(np.sort(np.abs(coef[:, 4:5]), axis=0))
    #print(np.where(np.abs(coef[:, 4:5]) > 0.01)[0])
    #print(np.where(np.abs(coef[:, 4:5]) > 0.01)[0].shape)
    print("COMPRESSED", k+1)
    #Cel -= 1
dim = 2
lowest = []
while len(lista) > 1:
    for i in range(len(lista)):
        for j in range(10):
            tren = 0
            for k in range(50):
                coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=tren)
                TRUE_ENERGY = np.sum((coef[:, 4:5] ** 2) ** (dim / 2) * feature_std[:, None] ** dim / goal_std[None, 4:5] ** dim)
                MX = TRUE_ENERGY / Cel
                error = Corr.Prediction_ofCoefficient(coeffiecient=coef[:, 4:5], dane_onlyX=pred[0], norm=False)
                error -= wynik[0][:, 4:5]
                error = np.mean(error ** 2) ** 0.5
                punish = 2 * np.log(MX)/MX * dim / (goal_std[4] ** dim * Cel)
                lahm = punish * feature_std[:, None] ** dim * ((coef[:, 4:5] ** 2) ** ((dim-2)/2))
                lahm = np.where(np.abs(lahm) < 10 ** 100, lahm, 10 ** 100 * np.sign(lahm))
                lahm[i+1,:] = 10**100
                lahm = np.eye(len(lahm)) * lahm
                tren = lahm * 0.1 + tren * 0.9
            Cel = TRUE_ENERGY
        print("CEL", TRUE_ENERGY , coef[:, 4:5].shape, lista[i], error)
        #lowest.append(error)
        lowest.append(error+Cel)
    lowest = np.array(lowest)
    mask = np.argsort(lowest)
    print("REMOVED", lista[mask[0]], lowest[mask[0]])
    mask = np.sort(mask[min_tres+1:])
    feature_std = feature_std[np.hstack((0, mask+1))]
    pred = [pre[:, mask] for pre in pred]
    lista = lista[mask]
    print(lista)
    lowest = []
    print("ENERGY SEARCH")
    for j in range(10):
        tren = 0
        for k in range(50):
            coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=tren)
            TRUE_ENERGY = np.sum((coef[:, 4:5] ** 2) ** (dim / 2) * feature_std[:, None] ** dim / goal_std[None, 4:5] ** dim)
            MX = TRUE_ENERGY / Cel
            error = Corr.Prediction_ofCoefficient(coeffiecient=coef[:, 4:5], dane_onlyX=pred[0], norm=False)
            error -= wynik[0][:, 4:5]
            error = np.mean(error ** 2) ** 0.5
            punish = 2 * np.log(MX)/MX * dim / (goal_std[4] ** dim * Cel)
            lahm = punish * feature_std[:, None] ** dim * ((coef[:, 4:5] ** 2) ** ((dim-2)/2))
            lahm = np.where(np.abs(lahm) < 10 ** 100, lahm, 10 ** 100 * np.sign(lahm))
            lahm = np.eye(len(lahm)) * lahm
            tren = lahm * 0.1 + tren * 0.9
        Cel = TRUE_ENERGY
        print("ENERGY SEARCH", Cel , error)
#Corr.Show_theThing(error)
#print("JESTE")
#Corr.Show_theThing(np.array(wynik[0])[:,4:5])
#kl"""
"""LAMBDA = [0.07287537161205404, 2.5660051811909175, 1.5355159078634903, 1.979455937332234, 2.2799226627389144, 2.61775525971217, 2.9707910388287457, 3.225370928277267, 3.417990118193088, 3.5872789687119004, 3.7514529315290157, 3.91612975904679, 4.084955705094684, 4.261247377790878, 4.447863437946252, 4.647859528172891, 4.864138653092768, 5.099387054844317, 5.35692668652944, 5.6407490041290735, 5.9544524389931315, 6.302108992311396, 6.690343554113489, 7.128459789792394, 7.627543442538678, 8.201529839482586, 8.869133505746426, 9.655902903381381, 10.598145761144629, 11.749641448239755, 13.192124720895906, 15.05619655637692, 17.57403046506692, 21.225598587041855, 27.227057052852725, 40.15626260394809]
LAMBDA = [0.0073627387199761305, 2.2915704202799647, 1.2956709657570973, 1.8286073471850095, 2.1937766209815863, 2.598434473863546, 2.9414601347127878, 3.164344239413132, 3.3316830572445846, 3.4813645178218358, 3.6283058598390876, 3.7791048468586426, 3.93842236192743, 4.10866767171411, 4.291353167504407, 4.488066334178304, 4.700183726740189, 4.92972694733594, 5.180033329864073, 
5.455597105994371, 5.761341533163366, 6.102539923382518, 6.485472073919847, 6.917591108344112, 7.408527021743158, 7.971423575870057, 8.623904732099914, 9.389038738437934, 10.2983594223826, 11.398711269637571, 12.763601829310883, 14.515303880713159, 16.876755046871075, 20.30582850773692, 25.939898070991347, 38.02295655277304]
target = (len(LAMBDA) - np.arange(len(LAMBDA))) * 0.01
print(target)
for i in range(1):
    pred = raw[:]
    #pred = wynik[:]
    #LAMBDA = [5,1,8,20,22,21,16,11]
    #LAMBDA = [0,0.08,11,12,10,9,8,8,7,7,7,7,6,6,6,6,5,5,5,5,5]
    #LAMBDA = [0,0.1,13,13,12,12,11,10,10,9,9,9,8,8,8,8,7,7,7,7,7]
    #LAMBDA = [0, 0.11,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
    #LAMBDA = [-3.4555015739616195, 0.09535186990306721, 0.850206330719623, 0.99116975103233, 0.8463256236197931]
    #LAMBDA = [-1.54482756330003, 2.033224133964543, 1.6720243427100616, 3.1970151959060393, 6.071726490472875]
    #target = [1,1,1,1,1]
    #goal  = -np.log(1/np.vstack(wynik) - 1)
    #LAMBDA = [0,0.08,10,10,11,11,9,8,8,7,7,6,7]
    #LAMBDA = [0,0.08,10.90,10.66,10.94,10.99,9.38,8.64,7.97,7.12,7.00,6.50,6.43,6.86,6.38,6.23,6.13,6]
    #LAMBDA = [0,0.02,1.48,1.27,1.30,1.85,3.49,7.32,14.46]
    #LAMBDA = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    yes = []
    yup = []
    dim = 2
    for L, t in zip(LAMBDA, target):
        #print(pred[-1].shape)
        coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=True, LAMBDA=L)
        #A = np.vstack(pred)
        #An = np.hstack((np.ones((A.shape[0],1)),A))
        #coef2 = np.linalg.pinv(An) @ -np.log(1/A - 1)
        #if coef2.shape == coef.shape:
        #    print(np.mean((coef - coef2)**2)**0.5)
        #print(np.mean(np.diag(coef2[1:,:])))
        #coef = coef2 * 0.1 + coef * 0.9
        coef_acc = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=L)
        acc = [Corr.Prediction_ofCoefficient(coeffiecient=coef_acc, dane_onlyX=pre, norm=False) for pre in pred]
        error = [np.mean(np.mean((acc[j][:, j*5:j*5+5] - wynik[j])**2, axis=0)**0.5/np.std(wynik[j],axis=0)) for j in range(len(acc))]
        error = np.mean(error)

        feature_std = np.hstack((0, np.std(np.vstack(pred), axis=0)))
        goal_std = np.hstack([np.std(wyn, axis=0) for wyn in wynik])
        TRUE_ENERGY = ((coef_acc ** 2) ** (dim / 2.0)) * (feature_std[:, None]**dim) / (goal_std[None, :]**dim)
        #print(np.sum(TRUE_ENERGY, axis=0))
        TRUE_ENERGY = np.mean(np.sum(TRUE_ENERGY, axis=0))
        TREU_LAMBDA = TRUE_ENERGY / t
        TREU_LAMBDA = 2 * np.log(TREU_LAMBDA)/TREU_LAMBDA * dim / (goal_std ** dim * t)
        TREU_LAMBDA = np.mean(TREU_LAMBDA)
        yes.append(float(TREU_LAMBDA))
        yup.append(float(TRUE_ENERGY))
        pred = [Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pre, norm=False) for pre in pred]
        #print(np.vstack(pred).shape)
        print("LAMBDA", L, "ERROR", error, "ENERGY", TRUE_ENERGY, TREU_LAMBDA)
        #print(L, np.std(np.std(np.vstack(pred), axis=0)), np.mean(np.std(np.vstack(pred), axis=0)), error,np.max(coef[1:,:]),np.mean(np.diag(coef[1:,:])),np.min(np.diag(coef[1:,:])), TRUE_ENERGY)
        #print(np.max(coef[1:,:]))
        
        pred = [1 / (1 + np.exp(-pre)) for pre in pred]
        #print(pred[-1][-1:,-5:])
        #print("HO", np.mean(np.max(np.abs(coef), axis=0)))
        
        
        #print((pred)[-1].shape)
        #print(len(pred))
    print()
    print("ERROR", error, "ENERGY", TRUE_ENERGY, TREU_LAMBDA)
    print(yes)
    LAMBDA = yes[:]
    #coef = Corr.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=False, LAMBDA=6)
    #pred = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pred[-1], norm=False)
    #print(len(pred))
    #input()
    #Corr.Show_theThing(pred[:, -5:])
    #print(np.mean(np.mean((pred[:, -5:] - wynik[-1])**2, axis=0)**0.5/np.std(wynik[-1],axis=0)))"""
Corr.Expand_system_experiment(raw[-1], wynik[-1])
gfd
test_out = np.array(wynik[-1][:, -1])
test_out = (test_out - test_out.mean()) / test_out.std()
test_out = 1 / (1 + np.exp(-test_out))
print(test_out.shape, raw[-1].shape[1])
raw = [(r - r.mean(axis=0)) / r.std(axis=0) for r in raw]
raw = [1 / (1 + np.exp(-r)) for r in raw]
token_id = []
for i in range(raw[-1].shape[1] - 1):
    for j in range(raw[-1].shape[1] - 1 - i):
        for k in (-1, 1):
            for l in (-1, 1):
                newone = np.zeros(raw[-1].shape[1])
                newone[i] = k
                newone[j+i+1] = l
                token_id.append(newone)
best_error = []
for token in token_id:
    pred_test = raw[-1] * token
    win_mask = (pred_test != 0)
    winn = pred_test * win_mask  
    winn = np.where(winn > 0, winn, winn + 1)
    winn = np.min(winn, axis=1)
    error = np.mean((np.log(1/winn-1) - np.log(1/test_out-1)) ** 2) ** 0.5
    best_error.append(error)
best_sorted = np.argsort(best_error)
print(np.sort(best_error)[:5])
input("Sorted errors")
print(best_sorted[:5])
feature_std = np.hstack((0, np.std(np.vstack(raw), axis=0)))
test_out = -np.log(1/test_out - 1)
Corr.Show_theThing(test_out[:, None])
for best in best_sorted[:100]:
    pred_test = raw[-1] * token_id[best]
    winn = np.array([[pre for pre in pred_t if pre != 0] for pred_t in pred_test])
    winn = np.where(winn > 0, winn, winn + 1)
    winn = np.min(winn, axis=1, keepdims=True)
    winn = -np.log(1 / winn - 1)
    goal_std = np.std(winn, axis=0)
    LAMBDA = 1.0378142897781557
    ENERGY = 1
    for i in range(1):
        LAMBDA = 1.0378142897781557
        for j in range(1000):
            coef = Corr.Coefficient(dane=raw[-1], wynik=winn, LAMBDA=LAMBDA, norm=False)
            #test = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pred[-1], norm=True)
            mx = np.sum((coef[:, :] ** 2) * (feature_std[:, None] **2) / (goal_std ** 2))
            stay = 4 * np.log(mx / ENERGY) / ((mx / ENERGY)) / (goal_std ** 2 * ENERGY)
            stay = stay[0]
            if (stay > LAMBDA * 1.0001) and (stay > LAMBDA):
                LAMBDA *= 1.0001
            elif (stay < LAMBDA * 0.9999) and (stay < LAMBDA):
                LAMBDA *= 0.9999
            else:
                LAMBDA = float(stay)
            print(LAMBDA, mx, coef.shape, stay, 1.026+j/100000, type(LAMBDA))
        ENERGY = mx
        print("ENERGY", ENERGY)
        input()
    test = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=raw[-1], norm=False)
    error = np.mean((test - test_out[:, None]) ** 2) ** 0.5
    error2 = np.mean((test - winn) ** 2)**0.5
    pred_noise = raw[-1] + feature_std[None, 1:] * np.random.normal(0, 1, raw[-1].shape)
    test_noise = Corr.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pred_noise, norm=False)
    noise_error = np.mean((test_noise - test_out[:, None]) ** 2) ** 0.5
    noise_error2 = np.mean((test_noise - winn) ** 2)**0.5
    #Corr.Show_theThing((test - winn)**2)
    print()
    #Corr.Show_theThing(coef)
    print("FINAL", ENERGY, LAMBDA)
    print(token_id[best])
    print("ERROR", error, error2)
    print("NOISE ERROR", noise_error, noise_error2)
    input("NEXT")
hj


"""lolly=[]
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
#[print(len(ana)) for ana in analyse]

#Corr.Correlation_betweenSession(raw, wynik, test=analyse[:-1])
#Corr.lowerdimension(dany=raw, wyniks=wynik, test=analyse[:-1])
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

#print(lol.shape)
name = ['Sezon 1 chapter 1','Sezon 1 chapter 2','Sezon 1 chapter 3','Sezon 1 chapter 4','Sezon 1 chapter 5','Sezon 2 chapter 1','Sezon 2 chapter 2','Sezon 2 chapter 3','Sezon 2 chapter 4','Sezon 3 chapter 1','Sezon 3 chapter 2', 'Sezon 3 chapter 3', '25 Sezon 9 chapter 1', '25 Sezon 10 chapter 1', '25 Sezon 10 chapter 2', '25 Sezon 10 chapter 3', '25 Sezon 10 chapter 4', '25 Sezon 11 chapter 1', '25 Sezon 11 chapter 2', '25 Sezon 11 chapter 3', '25 Sezon 11 chapter 4', '25 Sezon 12 chapter 1', '25 Sezon 12 chapter 2', f"Lumyn's Mixtape"]
name = name[:-1]
print(len(name))

analyse = analyse[:-1]
#name = name[-2:]
#for i in range(100):
#    k = 2 ** ((i)/50)
#    Corr.Prediction_Belonging(raw, wynik, test=analyse, norm=True, k = k)
#Corr.Show_theThing(np.argsort(-lol[-1], axis=1))
#d
pos = [0,5,9,12,17,21]
#season(["Maj","Czerwiec", "Lipiec","Październik", "Listopad"], pos, raw, wynik, analyse[4:], SpotifyAPI(), name)
amount = [45,35,26,21,17,13,9,8,6,5,3,3,3,1,2,0,1,1,0,0,0,1]
amount = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
#s
k = 50
print("SPECIAL SONGS")
special = []
force = np.arange(len(name))
force = np.exp(-force / 4)
print(force)
helper = np.zeros(force.shape)
while len(special) < k:
    kroki = (np.ones(helper.shape) - helper) / force
    special.append(int(np.argmin(kroki)))
    helper += force * np.min(kroki)
    helper[np.argmin(kroki)] = 0
print(special)
print(max(special))
input("SPECIAL")
matrix = Corr.Prediction_naSterydach(raw[:], wynik[:], norm=True, test=analyse[-len(name):])
distance = Corr.distance_matrix(matrix=np.vstack(matrix))
#Corr.Show_theThing(distance[:, -3:-2])
#Corr.Show_theThing(distance[np.argsort(distance[:, -4])[:100], -4:-3])
#input("DISTANCE")
#order = [np.argsort(distance[:, -4])]
#s.create_new_playlists(order, name, name_new_playlists=['LYRICS', 'SUPERIOR', 'SMART', 'INNI', '26 Sezon 1 Chapter 1'])
#t
order = None
for i in range(k):
    #if i == 0:
    #    order = Corr.ORDER(raw[:], wynik[:], analyse[-i-1:], 4, number_songs=amount[i], orde=order)
    #else:
    #    order += len(analyse[-i-1])
    population = sum([len(an) for an in analyse[-len(name):-special[i]-1]])
    #print("POP", population)
    population = list(range(population, population + sum([len(an) for an in analyse[-special[i]-1:]])))
    #print(population)
    order = Corr.ORDER(matrix=matrix, distance=distance, number_songs=1, orde=order, chosen=population)
    #order[:, -amount[i]:] += sum([len(an) for an in analyse[-len(name):-i-1]])
    print(order.shape[1])
    #print(len(order))
    #print(len(analyse[-special[i]-1]))
Corr.Show_theThing(order)
input()
for ode in order:
    Corr.Show_theThing(np.vstack(matrix)[ode, -5:])
    input()
name_new_playlists=['Moc', 'Intro', 'Lepszy', 'Pop', 'Yes']
s.create_new_playlists(order, name, name_new_playlists=name_new_playlists)
kl
for i in range(0):
    order = Corr.ORDER(raw[:], wynik[:], analyse[-len(name):], 4, number_songs=200)
    Corr.Show_theThing(order)
    #print(order.shape)
    input()
#order = Corr.GMM(np.vstack(lol), k_mean=11)
#order = Corr.K_mean(np.vstack(lol))
#name = [f"Lumyn's Mixtape"]
#[Corr.ORDER(raw, wynik, raw[-len(name):], si/100, number_songs=100) for si in range(2,3)]
#order = Corr.ORDER(raw[:], wynik[:], analyse[-len(name):], 4, number_songs=2000)
#input()
#order = Corr.ORDER(raw[:], wynik[:], analyse[-len(name):], 40, number_songs=1000, similar=order[0])

#input()
#print(len(order[-1]))
#s.create_new_playlists(order, name)
name_new_playlists=['Lym Gym', 'Lym Past', 'Lym Better', 'Lym idk', 'Lym Like']
name_new_playlists=[str(i+00) for i in range(len(order))]
#s.create_new_playlists(order, name, name_new_playlists=name_new_playlists)



