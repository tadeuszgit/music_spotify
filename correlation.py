import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class Correle:

    @staticmethod
    def Show_theThing(solution):
        for row in solution:
            txt = ""
            for dane in row:
                txt += str(dane) + ";"
            print(txt)
    
    @staticmethod
    def Coefficient(dane, wynik=None, number_wyniks = None, norm = False):
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
        if norm:
            x[0, :] -= total[:, -n:].mean(axis=0)
            x[:, :] /= total[:, -n:].std(axis=0)

        return x

    @staticmethod
    def Prediction_ofCoefficient(coeffiecient, dane_onlyX, match = False, norm=False):
        ones = np.ones((dane_onlyX.shape[0], 1))
        if match:
            ones = 1 - ones
            ones[0, 0] = 1
        dane = np.hstack((ones, dane_onlyX))
        #print(dane.shape, coeffiecient.shape)
        y_pred = dane @ coeffiecient
        if norm:
            y_pred = 1 / (1 + np.exp(-y_pred))
        return y_pred
    @staticmethod
    def Redundance(matrix, norm=True):
        corr = Correle.Correaltion(matrix=matrix)
        weight = 1/np.sum(corr ** 2, axis=0)
        if norm:
            weight /= weight.sum()
        return weight
    @staticmethod
    def ListRedundance(matrix):
        cand = list(range(matrix.shape[1]))
        lost = []
        while len(cand) > 0:
            scores = Correle.Redundance(matrix=matrix[:, cand])
            lost.append(cand.pop(np.argmin(scores)))
            print(lost[-1])
        #print(Correle.Redundance(matrix=matrix[:, lost], norm=False))
        return lost
    @staticmethod
    def Coefficient_for_all_dane(dany, wyniks, unsafe = False, norm = False, save=10):
        if unsafe:
            norm = False
        coefiecients = [Correle.Coefficient(dane, wynik, norm=norm) for dane, wynik in zip(dany, wyniks)]
        
        yy_pred = []
        for dane in dany:
            yy_pred.append([Correle.Prediction_ofCoefficient(coeffiecient, dane, norm=norm) for coeffiecient in coefiecients])
            #print(np.hstack(yy_pred[-1]).shape)
        
        if unsafe:
            prepe = np.hstack(coefiecients)
            return prepe
        ultra_coef = []
        norm_coef = np.hstack(coefiecients)
        if norm:
            #optimal = Correle.ListRedundance(np.vstack([np.hstack(pred) for pred in yy_pred]))
            #print(optimal)
            #print("USE IT!!!")
            optimal = [43, 8, 63, 5, 68, 6, 42, 79, 61, 78, 30, 53, 39, 56, 40, 7, 22, 57, 19, 45, 77, 20, 38, 13, 0, 54, 65, 17, 71, 36, 58, 52, 83, 46, 25, 81, 59, 21, 74, 67, 15, 47, 12, 11, 34, 31, 3, 76, 72, 29, 51, 50, 23, 27, 75, 55, 9, 37, 64, 48, 80, 26, 69, 41, 16, 33, 14, 73, 4, 44, 24, 84, 35, 18, 28, 2, 66, 60, 1, 70, 82, 62, 32, 10, 49]
            optimal = list(range(85))
            norm_coef = norm_coef[:, optimal[-save:]]
        #print(norm_coef.shape)
        #ultra_pred = []
        for i, dane in enumerate(dany):
            dane_x = np.hstack((yy_pred[i][:i]+ yy_pred[i][i+1:]))
            number = sum([len(yy_pred[i][j][-1]) for j in range(len(dany)) if j < i]) + 1
            test = []
            for j in range(len(coefiecients)):
                if j != i:
                    test.append(coefiecients[j])
                else:
                    test.append(np.zeros(coefiecients[j].shape))
            test = np.hstack(test)
            if norm:
                dane_x = np.hstack(yy_pred[i])[:, optimal[-save:]]
                #print("HERE")
                #print(dane_x.shape)
                #input()
            mega_coeffiecient = Correle.Coefficient(dane_x, wynik=wyniks[i])
            #print(mega_coeffiecient.shape)
            if norm:
                ultra_coef.append(mega_coeffiecient)
            else:
                test2 = np.vstack([mega_coeffiecient[:number,:], np.zeros((len(yy_pred[i][i][-1]), len(yy_pred[i][i][-1]))), mega_coeffiecient[number:,:]])
                ultra_coef.append(Correle.Prediction_ofCoefficient(coeffiecient=test2, dane_onlyX=test, match = True))
        ultra_coef = np.hstack(ultra_coef)
        if norm:
            return norm_coef, ultra_coef
        return ultra_coef
    @staticmethod
    def Prediction_ofNorm(dany, wynik, test=None, save=9):
        if test is None:
                test = dany
        norm, coef = Correle.Coefficient_for_all_dane(dany=dany,wyniks=wynik, norm=True, save=save)
        Correle.Show_theThing(coef[:, -5:])
        print("THE MYSTERY")
        input()
        hidden = [Correle.Prediction_ofCoefficient(dane_onlyX=dan, coeffiecient=norm, norm=True) for dan in test]
        result = [Correle.Prediction_ofCoefficient(dane_onlyX=dan, coeffiecient=coef) for dan in hidden]
        return result
    @staticmethod
    def Prediction_ofBIGCoefficient(dany, wynik, test=None, unsafe=False):
        coef = Correle.Coefficient_for_all_dane(dany=dany, wyniks=wynik, unsafe=unsafe)
        
        #print(coef.shape)
        #test = np.vstack(test)
        #print(test.shape)
        if test:
            pred = Correle.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=np.vstack(test))
        else:
            pred = [Correle.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=dan) for dan in dany]
        return pred
    
    @staticmethod
    def Correaltion(matrix, axis=1):
        #Correle.Show_theThing(matrix.std(axis=0, keepdims=True))
        #print("here")
        #Correle.Show_theThing(matrix[:, -10:-5])
        #input()
        if axis == 1:
            p = [[np.mean((matrix[:, x] - matrix[:, x].mean()) * (matrix[:, y] - matrix[:, y].mean()))/(matrix[:, x].std()*matrix[:, y].std()) for x in range(matrix.shape[1])] for y in range(matrix.shape[1])]
        else:
            p = [[np.mean((matrix[x, :] - matrix[x, :].mean()) * (matrix[y, :] - matrix[y, :].mean()))/(matrix[x, :].std()*matrix[y, :].std()) for x in range(matrix.shape[0])] for y in range(matrix.shape[0])]
        return np.array(p)
    @staticmethod
    def distance_matrix(dany=None, wyniks=None, matrix = None, unsafe=False):
        if matrix is None:
            matrix = Correle.Prediction_ofBIGCoefficient(dany=dany, wynik=wyniks, test=dany, unsafe=unsafe)
        corr = Correle.Correaltion(matrix=matrix)
        weight = 1/np.sum(corr**2, axis=0)
        weight /= weight.sum()
        distance = (matrix[:, None, :] - matrix[None, :, :]) ** 2 @ weight
        distance = distance ** 0.5
        return distance
    @staticmethod
    def ORDER(dany, wyniks, test, SIGMA, number_songs = 100):
        matrix = Correle.Prediction_ofNorm(dany, wyniks, test)
        matrix = np.vstack(matrix)
        #if matrix.shape[0] < number_songs:
        #    number_songs = matrix.shape[0]
        #Correle.Show_theThing(matrix[:, -wyniks[-1].shape[1]:])
        #print(matrix[:, -wyniks[-1].shape[1]:].shape)
        #print(dany.shape)
        #input()
        #corr = Correle.Correaltion(matrix)
        #weight = 1/np.sum(corr**2, axis=0)
        #weight /= weight.sum()
        #print(weight/weight.mean())
        #print(weight.min()/weight.mean(), weight.max()/weight.mean())
        matrix = (matrix - matrix.mean(axis=0))/matrix.std(axis=0)
        matrix = 1 / (1 + np.exp(-matrix))
        """TRY TO FIND THE SIGMA male"""
        #distance = (matrix[:, None, :] - matrix[None, :, :]) ** 2 @ weight
        #distance = distance ** 0.5
        distance = Correle.distance_matrix(matrix=matrix)
        winners = []
        uniq = 5
        #power = np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0) / np.mean(np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0))
        power = np.exp(-distance**2/SIGMA**2)
        for i in range(uniq):
            gmatrix = matrix[:, -uniq+i]
            gpower = power * gmatrix[:, None]
            winner = [np.argmax(gmatrix)]
            while len(winner) < number_songs:
                luck = gmatrix / gpower[winner, :].mean(axis=0)
                idx = np.argmax(luck)
                winner.append(idx)
            winners.append(winner)
            #winners.append(np.argsort(1-gmatrix))
            #print(winner)
        #input()
        """for i in range(uniq):
            winner = []
            gdistance = distance[:, :]
            gmatrix = matrix[:, :]
            ingame = np.array(list(range(distance.shape[0])))
            while ingame.shape[0] > 0:
                power = np.sum(np.exp(-gdistance ** 2 / SIGMA ** 2), axis=0) / np.mean(np.sum(np.exp(-gdistance ** 2 / SIGMA ** 2), axis=0))
                gpower = np.exp(gmatrix[:, -uniq+i]) / power
                idx = np.argmax(gpower)
                winner.append(ingame[idx])
                #print(winner[-1]+2, np.max(gpower), power[idx], gmatrix[idx, uniq+i])
                ingame = np.delete(ingame, idx)
                gdistance = distance[ingame, :]
                gdistance = gdistance[:, ingame]
                gmatrix = matrix[ingame, :]
            winners.append(winner[:number_songs])
            print(len(winner))"""
        power = np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0) / np.mean(np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0))
        #print(winners[-1])
        #Correle.Show_theThing(power[:, None])
        print(SIGMA, np.sum(power**2)/matrix.shape[0]-1)
        return winners
    @staticmethod
    def Correlation_betweenSession(dany, wyniks):
        matrix = Correle.Prediction_ofNorm(dany=dany, wynik=wyniks, save=10)
        [print(matri.shape) for matri in matrix]
        matrix = np.array([matri.mean(axis=0) for matri in matrix])
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        dmatrix = Correle.distance_matrix(matrix=matrix)
        Correle.Show_theThing(dmatrix)
        print(dmatrix.shape)
        input()
        Correle.Show_theThing(matrix[:, 4::5])
        print(matrix.shape)
        input()
        matrix = Correle.Correaltion(matrix=matrix, axis=0)
        Correle.Show_theThing(matrix)
        print(matrix.shape)
    @staticmethod
    def lowerdimension(dany = None, wyniks = None, distance = None):
        matrix = Correle.Prediction_ofNorm(dany=dany, wynik=wyniks, save=10)
        matrix = np.array([matri.mean(axis=0) for matri in matrix])
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        matrix = 1 / (1 + np.exp(-matrix))
        if distance is None:
            distance = Correle.distance_matrix(matrix=matrix)
            #distance = Correle.distance_matrix(dany=dany, wyniks=wyniks)
        n = distance.shape[0]
        J = np.eye(n) - np.ones((n,n)) / n
        D2 = distance ** 2
        B = -0.5 * J @ D2 @ J
        values, vectors = np.linalg.eig(B)

        idx = np.argsort(values)[::-1]
        values = values[idx]
        vectors = vectors[:, idx]
        pos = (np.isreal(values)) & (values > 0)
        values = values[pos].real
        vectors = vectors[:, pos].real

        window = 200

        x = vectors @ np.diag(np.sqrt(values))
        #x = np.array([np.mean(x[i:i+window],axis=0) for i in range(x.shape[0] - window)])
        
        cmap = plt.get_cmap('plasma')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2, 0], x[i:i+2, 1], x[i:i+2, 2], color=cmap(i/(x.shape[0] - 2)))

        #plt.plot(x[:, 0], x[:, 1], cmap='plasma', c = np.linspace(0,1, x.shape[0]))
        plt.show()
        print(values)
    @staticmethod
    def Check_mass_correlation(dany, wyniks):
        ultra = Correle.Coefficient_for_all_dane(dany, wyniks)
        uniq = 0
        i = 0
        diff = []
        for wynik in wyniks:
            ses_diff = (ultra[i:i+wynik.shape[0], uniq:uniq+wynik.shape[1]] - wynik) ** 2
            diff.append(ses_diff)
            i += wynik.shape[0]
            uniq += wynik.shape[1]
            print(type(ses_diff))
            Correle.Show_theThing(ses_diff.mean(axis=1, keepdims=True) ** 0.5)
            #input()
        error = [float(ses_diff.mean() ** 0.5) for ses_diff in diff]
        print()
        print(error)

def test_accuracy(max_groups = 200, period = 100, members = 5, atribu = 3, umie = 2):
    for i in range(10, max_groups):
        anomalia = []
        comp_anomalia = []
        for j in range(period):
            dany = np.random.random((i, members, atribu))
            wynik = np.random.random((i, members, umie))
            p = Correle.Coefficient_for_all_dane(dany, wynik)
            comp_p = Correle.Coefficient_for_all_dane(dany, wynik, unsafe=True)
            #c.Show_theThing(np.round((comp_p - p) * 100))
            #input()
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
            #print(np.sum(p>1), np.sum(p<0))
            #print(np.sum(comp_p>1), np.sum(comp_p<0))
            #print(np.sum(p**2), np.sum(comp_p**2))
            #input()
            total_anomalie = total_anomalie / (p.shape[0] * p.shape[1] - i * members * umie)
            comp_total_anomalie = comp_total_anomalie / (p.shape[0] * p.shape[1] - i * members * umie)
            #print(total_anomalie, comp_total_anomalie)
            
            anomalia.append(total_anomalie)
            comp_anomalia.append(comp_total_anomalie)
        anomalia = np.array(anomalia)
        comp_anomalia = np.array(comp_anomalia)
        print(f"{i}: {np.median(anomalia) * 100:.7f}% {np.median(comp_anomalia) * 100:.7f}%")
        #input()