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
    
    def Correlation_for_all_dane(self, dany, wyniks):
        coefiecients = [self.Correlation(dane, wynik) for dane, wynik in zip(dany, wyniks)]
        
        yy_pred = []
        for dane in dany:
            yy_pred.append(([self.Prediction_ofCorrelation(coeffiecient, dane) for coeffiecient in coefiecients]))
        ultra_pred = []
        for i, dane in enumerate(dany):
            #print(i)
            dane_x = np.hstack(yy_pred[i][:i]+yy_pred[i][i+1:])
            #dane_x = np.hstack((yy_pred[i][:i],yy_pred[i][i+1:]))
            mega_coeffiecient = self.Correlation(dane_x, wynik=wyniks[i])
            dane_xx = [np.hstack(yy_pred[k][:i]+yy_pred[k][i+1:]) for k in range(len(yy_pred))]
            #dane_xx = [np.hstack((np.hstack(yy_pred[k][:i]),np.hstack(yy_pred[k][i+1:]))) for k in range(len(yy_pred))]
            pred = np.vstack([self.Prediction_ofCorrelation(coeffiecient=mega_coeffiecient, dane_onlyX=danu) for danu in dane_xx])
            ultra_pred.append(pred)
        ultra_pred = np.hstack(ultra_pred)

        return ultra_pred
    

def test_accuracy(max_groups = 200, period = 100, members = 5, atribu = 3, umie = 2):
    c = Correle()
    for i in range(9, max_groups):
        anomalia = []
        for j in range(period):
            dany = np.random.random((i, members, atribu))
            wynik = np.random.random((i, members, umie))
            p = c.Correlation_for_all_dane(dany, wynik)
            total_anomalie = np.sum(p > 1) + np.sum(p < 0)
            total_anomalie = total_anomalie / (p.shape[0] * p.shape[1] - i * members * umie)
            anomalia.append(total_anomalie)
        anomalia = np.array(anomalia)
        print(f"{i}: {np.median(anomalia) * 100:.7f}% {np.mean(anomalia) * 100:.7f}%")
        #input()

test_accuracy(period=100, members=100, atribu = 12, umie = 4)
print("deon")
input()
dany = np.random.random((100,5,3))
wynik = np.random.random((100,5,2))
c = Correle()
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