import numpy as np

class Correle:
    def Show_theThing(self, solution):
        for row in solution:
            txt = ""
            for dane in row:
                txt += str(dane) + ";"
            print(txt)

    def Correlation(self, dane, wynik=None, number_wyniks = None):
        #CHECK FOR L2 REGULATION!!!!!!
        if wynik is None:
            total = dane[:]
            n = 1
            if number_wyniks is not None:
                n = number_wyniks
            m = len(total[-1]) - n
        else:
            m = len(dane[-1])
            n = len(wynik[-1])
            total = np.hstack((dane, wynik))

        matrix_one = np.hstack((np.ones((total.shape[0], 1)), total))

        mat = matrix_one[:, :m+1].T @ matrix_one
        A = mat[:,:m+1]
        b = mat[:, m+1:]
        x = np.linalg.solve(A, b)

        return x
    
    def Prediction_ofCorrelation(self, coeffiecient, dane_onlyX):
        ones = np.ones((dane_onlyX.shape[0], 1))
        dane = np.hstack((ones, dane_onlyX))
        y_pred = dane @ coeffiecient

        return y_pred
    
    def Correlation_for_all_dane(self, dany, wyniks, unsafe = False):
        coefiecients = [self.Correlation(dane, wynik) for dane, wynik in zip(dany, wyniks)]
        
        yy_pred = []
        for dane in dany:
            yy_pred.append(([self.Prediction_ofCorrelation(coeffiecient, dane) for coeffiecient in coefiecients]))
        if unsafe:
            prepe = np.vstack(np.array([np.hstack(prep) for prep in yy_pred]))
            return prepe
        ultra_pred = []
        for i, dane in enumerate(dany):
            #print(i)
            dane_x = np.hstack(yy_pred[i][:i]+yy_pred[i][i+1:])
            #dane_x = np.hstack((yy_pred[i][:i],yy_pred[i][i+1:]))
            mega_coeffiecient = self.Correlation(dane_x, wynik=wyniks[i])
            dane_xx = [np.hstack(yy_pred[k][:i]+yy_pred[k][i+1:]) for k in range(len(yy_pred))]
            #dane_xx = [np.hstack((np.hstack(yy_pred[k][:i]),np.hstack(yy_pred[k][i+1:]))) for k in range(len(yy_pred))]
            pred = np.hstack([self.Prediction_ofCorrelation(coeffiecient=mega_coeffiecient, dane_onlyX=danu) for danu in dane_xx])
            ultra_pred.append(pred)
        ultra_pred = np.vstack(ultra_pred)

        return ultra_pred


def test_accuracy(max_groups = 200, period = 100, members = 5, atribu = 3, umie = 2):
    c = Correle()
    for i in range(100, max_groups):
        anomalia = []
        comp_anomalia = []
        for j in range(period):
            dany = np.random.random((i, members, atribu))
            wynik = np.random.random((i, members, umie))
            p = c.Correlation_for_all_dane(dany, wynik)
            comp_p = c.Correlation_for_all_dane(dany, wynik, unsafe=True)
            c.Show_theThing(np.round((comp_p - p) * 100))
            input()
            #testt = np.where(p > 1, 1, np.where(p < 0, -1, 0))
            #c.Show_theThing(testt)
            #input()
            
            #testt = np.where(comp_p > 1, 1, np.where(comp_p < 0, -1, 0))
            #c.Show_theThing(testt)
            #input()
            #std_feat = np.round(np.std(p, keepdims=True, axis=0)*100) - np.round(np.std(comp_p, keepdims=True, axis=0)*100)
            #c.Show_theThing(std_feat)
            #print(std_feat.mean())
            #input()
            total_anomalie = np.sum(p > 1) + np.sum(p < 0)
            comp_total_anomalie = np.sum(comp_p > 1) + np.sum(comp_p < 0)
            #print("HERE")
            print(np.sum(p>1), np.sum(p<0))
            print(np.sum(comp_p>1), np.sum(comp_p<0))
            print(np.sum(p**2), np.sum(comp_p**2))
            input()
            total_anomalie = total_anomalie / (p.shape[0] * p.shape[1] - i * members * umie)
            comp_total_anomalie = comp_total_anomalie / (p.shape[0] * p.shape[1] - i * members * umie)
            print(total_anomalie, comp_total_anomalie)
            
            anomalia.append(total_anomalie)
            comp_anomalia.append(comp_total_anomalie)
        anomalia = np.array(anomalia)
        comp_anomalia = np.array(comp_anomalia)
        print(f"{i}: {np.mean(anomalia) * 100:.7f}% {np.mean(comp_anomalia) * 100:.7f}%")
        input()



test_accuracy(period=10, members=5, atribu = 3, umie = 2)
print("deon")
input()


print("DP")
p = c.Correlation_for_all_dane(dany, wynik)
print()
#c.Show_theThing(p)
print()
w = np.zeros((wynik.shape))
for g in range(wynik.shape[0]):
    w[g] = wynik[g] - p[g*wynik.shape[1]:g*wynik.shape[1]+wynik.shape[1], g*wynik.shape[2]:g*wynik.shape[2]+wynik.shape[2]]
    #w[g] = w[g] / wynik[g]
c.Show_theThing(np.hstack(wynik))
print()
print(p.shape)
print((np.sum(p > 1) + np.sum(p < 0))/p.shape[0]/p.shape[1])