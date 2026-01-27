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
    def Coefficient_APPROX(dane, wynik=None, number_wyniks = None, norm = False, LAMBDA = None):
        if LAMBDA is None:
            LAMBDA = 1
        if wynik is None:
            total = dane[:]
            n = 1
            if number_wyniks is not None:
                n = number_wyniks
            m = len(total[-1]) - n
        else:
            m = len(dane[-1])
            wynik = wynik[:, -1:]
            wynik = (wynik - np.mean(wynik)) / np.std(wynik)
            print(wynik.shape)
            n = len(wynik[-1])
            total = np.hstack((dane, wynik))

        matrix_one = np.hstack((np.ones((total.shape[0], 1)), total))
        mat = matrix_one[:, :m+1].T @ matrix_one
        A = mat[:, :m+1]
        Arc = LAMBDA * np.eye(A.shape[0]) * matrix_one[:, :m+1].std(axis=0, keepdims=True) ** 2
        Correle.Show_theThing(Arc)
        b = mat[:, m+1:]
        x = np.linalg.solve(A+Arc, b)
        for i in range(8800):
            #print(x.shape)
            #print((x.T * matrix_one[:, :m+1].std(axis=0, keepdims=True) @ x).shape, x.shape, "?")
            k = np.trace((x.T*matrix_one[:, :m+1].std(axis=0)** 2) @ x)
            k = np.trace(x.T @ x)
            g = matrix_one[:, :m+1] @ x - wynik
            #print(x.shape, k.shape, g.shape, wynik.shape, b.shape)
            g = np.trace(g.T @ g)
            #print(m+1, matrix_one[:, :m+1].shape)
            #print(g.shape, k.shape, A.shape)
            AA = 2 * g / k / np.log(k) * np.eye(A.shape[0]) + A

            bb = b
            #x_new = np.linalg.solve(AA, bb)
            #print(k, np.log(k), 2 * g / k / np.log(k), (g/matrix_one.shape[0]) ** 0.5, np.std(total[:, -1:]))
            slope = g * 2 * np.log(k) * x+ np.log(k) ** 2 * matrix_one[:, :m+1].T @ (matrix_one[:, :m+1] @ x - wynik)
            error = g * np.log(k) **2
            print(np.sum(slope), k, (g/matrix_one.shape[0]) ** 0.5, error )
            #print(x.T @ x)
            #print(k, np.max(np.abs((x_new - x)/x)), np.mean(np.abs((x_new - x)/x)))
            x -= slope * 0.000001
            #Correle.Show_theThing(x)
            #k = 2 * np.mean(np.log(np.sum(x**2, axis=0))/np.sum(x**2, axis=0))
            #x = np.linalg.solve(A + Arc, b)
            #print(np.mean(matrix_one[:, :m+1]@x, axis=0).shape, x.shape, wynik.shape)
            #print("K", np.log(np.sum(x**2, axis=0))**2, np.sum((matrix_one[:, :m+1] @ x - wynik) ** 2/np.std(wynik, axis=0)**2, axis=0))
            #print("ERRPR", np.sum(np.log(np.sum(x**2, axis=0))**2) + np.sum((matrix_one[:, :m+1] @ x - wynik) ** 2/np.std(wynik, axis=0)**2))
        Correle.Show_theThing(x)
        if norm:
            x[0, :] -= total[:, -n:].mean(axis=0)
            x[:, :] /= total[:, -n:].std(axis=0)
        return x

    @staticmethod
    #18.6
    def Coefficient(dane, wynik=None, number_wyniks = None, norm = False, LAMBDA = 1):

        if LAMBDA is None:
            LAMBDA = 10
            print(":P:")
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
        if type(LAMBDA) == int or type(LAMBDA) == float:
            A += LAMBDA * np.eye(A.shape[0]) * matrix_one[:, :m+1].std(axis=0, keepdims=True) ** 2
        else:
            A += LAMBDA
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
    def Coefficient_for_all_dane(dany, wyniks, unsafe = False, norm = False, save=10, LAMBDA=None):
        #if unsafe:
        #    norm = False
        coefiecients = [Correle.Coefficient(dane, wynik, norm=norm, LAMBDA=LAMBDA) for dane, wynik in zip(dany, wyniks)]
        if unsafe:
            prepe = np.hstack(coefiecients)
            return prepe
        yy_pred = []
        for dane in dany:
            yy_pred.append([Correle.Prediction_ofCoefficient(coeffiecient, dane, norm=norm) for coeffiecient in coefiecients])
            #print(np.hstack(yy_pred[-1]).shape)
        
        
        ultra_coef = []
        norm_coef = np.hstack(coefiecients)
        if norm:
            #optimal = Correle.ListRedundance(np.vstack([np.hstack(pred) for pred in yy_pred]))
            #print(optimal)
            #print("USE IT!!!")
            optimal = [43, 8, 63, 5, 68, 6, 42, 79, 61, 78, 30, 53, 39, 56, 40, 7, 22, 57, 19, 45, 77, 20, 38, 13, 0, 54, 65, 17, 71, 36, 58, 52, 83, 46, 25, 81, 59, 21, 74, 67, 15, 47, 12, 11, 34, 31, 3, 76, 72, 29, 51, 50, 23, 27, 75, 55, 9, 37, 64, 48, 80, 26, 69, 41, 16, 33, 14, 73, 4, 44, 24, 84, 35, 18, 28, 2, 66, 60, 1, 70, 82, 62, 32, 10, 49]
            #optimal = list(range(85))
            #print(len(optimal))
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
            mega_coeffiecient = Correle.Coefficient(dane_x, wynik=wyniks[i], LAMBDA=LAMBDA)
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
    def Prediction_ofNorm(dany, wynik, test=None, save=85 , LAMBDA=None):
        if test is None:
                test = dany
        norm, coef = Correle.Coefficient_for_all_dane(dany=dany,wyniks=wynik, norm=True, save=save, LAMBDA=LAMBDA)
        #Correle.Show_theThing(coef[:, -5:])
        #print("THE MYSTERY")
        #input()
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
    def distance_matrix(dany=None, wyniks=None, matrix = None, unsafe=False, norm=False):
        if matrix is None:
            matrix = Correle.Prediction_ofBIGCoefficient(dany=dany, wynik=wyniks, test=dany, unsafe=unsafe, norm=True)
        corr = Correle.Correaltion(matrix=matrix)
        weight = 1/np.sum(corr**2, axis=0)
        weight /= weight.sum()
        weight *= weight.shape
        matri = matrix - matrix.mean(axis=0)
        matri *= weight[None, :]
        distance = matri @ matri.T
        distance = 1 - distance / (np.diag(distance)[:, None] @ np.diag(distance)[None, :]) ** 0.5
        print(matrix.shape)
        print(distance.shape)
        #distance = (matrix[:, None, :] - matrix[None, :, :]) ** 2 @ weight
        #distance = distance ** 0.5
        return distance
    #FULL_ANALYSE HAVE TO BE GIVE
    def Prediction_Belonging(pred, wynik, LAMBDA1 = None, LAMBDA2 = None, test = None, norm = False, size = None, k = 1):
        matrix = Correle.Prediction_naSterydach(pred=pred, wynik=wynik, LAMBDA=LAMBDA1, test=test, norm = True)
        wynik2 = []
        for i in range(len(matrix)):
            mat = np.zeros((matrix[i].shape[0], len(matrix)))
            mat[:, i] = 1
            wynik2.append(mat)
        if LAMBDA2 is None:
            LAMBDA2 = [0]
        if size is None:
            size = len(test)
        test = matrix[-size:]
        matrix = [np.vstack(matrix)]
        wynik2 = [np.vstack(wynik2)]
        #print(wynik2[-1][-2])
        for i in range(200):
            matrix = Correle.Prediction_naSterydach(pred=matrix, wynik=wynik2, LAMBDA=LAMBDA2, norm = norm)
            wynik2 = [((mate - np.min(mate, axis=1)[:, None])/(np.max(mate, axis=1)[:, None] - np.min(mate, axis=1)[:, None])) ** k for mate in matrix]
        #Correle.Show_theThing(wynik2[-1][-80:, :])
        print(k, np.mean(wynik2[-1]), np.std(np.mean(wynik2[-1], axis=1)), np.min(np.sum(wynik2[-1], axis=1)))
        return matrix
    @staticmethod
    def Prediction_naSterydach(pred, wynik, LAMBDA = None, test = None, norm=False):
        #LAMBDA = LAMBDA[:]
        if LAMBDA is None:
            LAMBDA = [0, 0.11,14,15]
            #LAMBDA = [0,0.1,13,13,12,12,11,10,10,9,9,9,8,8,8,8,7,7,7,7,7]
            LAMBDA = [0.0073627387199761305, 2.2915704202799647, 1.2956709657570973, 1.8286073471850095, 2.1937766209815863, 2.598434473863546, 2.9414601347127878, 3.164344239413132, 3.3316830572445846, 3.4813645178218358, 3.6283058598390876, 3.7791048468586426, 3.93842236192743, 4.10866767171411, 4.291353167504407, 4.488066334178304, 4.700183726740189, 4.92972694733594, 5.180033329864073, 5.455597105994371, 5.761341533163366, 6.102539923382518, 6.485472073919847, 6.917591108344112, 7.408527021743158, 7.971423575870057, 8.623904732099914, 9.389038738437934, 10.2983594223826, 11.398711269637571, 12.763601829310883, 14.515303880713159, 16.876755046871075, 20.30582850773692, 25.939898070991347, 38.02295655277304]
            print("DEFAULT LAMBDA USED")
        if test is None:
            test = pred[:]
        for L in LAMBDA[:-1]:
            coef = Correle.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=True, LAMBDA=L)
            pred = [Correle.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=pre, norm=True) for pre in pred]
            test = [Correle.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=tes, norm=True) for tes in test]
            #print(np.vstack(pred).shape)
            #print(L, np.max(np.std(np.vstack(pred), axis=0)), np.mean(np.std(np.vstack(pred), axis=0)))
            #pred = [1 / (1 + np.exp(-pre)) for pre in pred]
            #Correle.Show_theThing(coef[:,-5:])
            #print("loly")
            #input()
        coef = Correle.Coefficient_for_all_dane(dany=pred, wyniks=wynik, unsafe=True, norm=norm, LAMBDA=LAMBDA[-1])
        #print(coef.shape, test[0].shape)
        pred = [Correle.Prediction_ofCoefficient(coeffiecient=coef, dane_onlyX=tes, norm=norm) for tes in test]
        #Correle.Show_theThing(coef[np.argsort(coef[:, -1:].T).T,-1])
        #print("loly end", LAMBDA[-1])
        #print("POWER")
        #Correle.Show_theThing(np.argsort(coef[1:, -1:].T))
        #print(np.max(coef[1:,:], axis=1))
        #Correle.Show_theThing(np.round(coef*100)/100)
        #input()
        don = np.argsort(coef[1:, :].T)
        Correle.Show_theThing(don[:, -10:])
        Correle.Show_theThing(coef[:, -1:])
        #print(coef.shape)
        #print(np.sum(coef[:, -1:] ** 2))
        #input()
        Correle.Show_theThing(coef[:, -5:])
        print("FINAL COEF")
        input()
        coef = coef[1:, 4::5][:, -5:]
        #coef = (coef - coef.mean(axis=0, keepdims=True))/coef.std(axis=0, keepdims=True)
        corr = Correle.Correaltion(matrix=coef, axis=1)
        #Correle.Show_theThing(pred[-1][-5:, -5:])
        #Correle.Show_theThing(corr)
        #print(corr.shape, coef.shape)
        #print("WHO IS WHO!")
        #input()
        return pred
    @staticmethod
    def ORDER_better(dany, wyniks, test, SIGMA, number_songs = 100):
        matrix = Correle.Prediction_naSterydach(dany, wyniks, test = test, norm=True)
        matrix = np.vstack(matrix)
        distance = Correle.distance_matrix(matrix=matrix)
        SIGMA = np.mean(np.mean(distance ** 2, axis=0)**0.5)/SIGMA
        POWER = np.exp(-distance**2/SIGMA**2)

        uniq = 5
        winners = []
        for i in range(uniq):
            score = matrix[:, -uniq+i]
            winner = [np.argmax(score)]
            dense = POWER.sum(axis=1)
            dense  = np.where(dense > 0.1, dense, 0.1)
            true_mean = np.sum(score/dense) / np.sum(1/dense)
            true_std = np.sqrt(np.sum((score - true_mean) ** 2/dense) / np.sum(1/dense))
            true_score = (score - true_mean) / true_std
            true_score = 1 / (1 + np.exp(-true_score))
            true_score = true_score / dense
            true_goal = true_score / true_score.sum()
            counter = np.zeros(true_goal.shape)
            counter[np.argmax(score)] += 1
            rel_counter = counter / counter.sum()
            while len(winner) < number_songs:
                true_power = POWER[:, winner[-1:]]
                dense = true_power.sum(axis=1)
                dense  = np.where(dense > 0.1, dense, 0.1)
                true_mean = np.sum(score/dense) / np.sum(1/dense)
                true_std = np.sqrt(np.sum((score - true_mean) ** 2/dense) / np.sum(1/dense))
                true_score = (score - true_mean) / true_std
                true_score = 1 / (1 + np.exp(-true_score))
                true_score = true_score / dense
                true_score = true_score / true_score.sum()
                counter[np.argmax(true_score * np.where(rel_counter < true_goal, 1, 0))] += 1
                rel_counter = counter / counter.sum()
                
                #win = np.argmax(true_score)
                #print(win, np.max(dense), np.min(true_power.sum(axis=1)), true_power.sum(axis=1)[win])
                #Correle.Show_theThing(matrix[[win], -5:])
                #input()
                winner.append(np.argmax(true_score * np.where(rel_counter < true_goal, 1, 0)))
                print(POWER[winner[-1], winner[-2]])
            print()
            Correle.Show_theThing(np.vstack((true_score * 1000, true_goal * 1000, counter)).T)
            input()
            #print()
            #Correle.Show_theThing(matrix[winner[:1], -5:])
            #Correle.Show_theThing(matrix[winner, -5:].mean(axis=0, keepdims=True))
            #Correle.Show_theThing(matrix[winner, -5:].std(axis=0, keepdims=True))
            winners.append(winner)
        return winners
    @staticmethod
    def ORDER_TheBEST(dany, wyniks, test):
        matrix = Correle.Prediction_naSterydach(dany, wyniks, test = test, norm=True)
        matrix = np.vstack(matrix)
        gmatrix = matrix[:, 4::5]
        gmatrix = np.prod(gmatrix, axis=1)
        return [np.argsort(1-gmatrix)]
    @staticmethod
    def order(matrix, distance, SIGMA = 4, number_songs=100, orde = None, chosen = None):
        SIGMA = np.mean(np.mean(distance ** 2, axis=0)**0.5)/SIGMA
        power = np.exp(-distance**2/SIGMA**2)
        winners = []
        if orde is not None:
            number_songs += len(orde[-1])
        for i in range(matrix.shape[1]):
            gmatrix = matrix[:, i]
            gpower = power * gmatrix[:, None]
            #print(gpower.shape)
            #input()
            if orde is None:
                winner = [np.argmax(gmatrix[chosen]) + chosen[0]] if chosen is not None else [np.argmax(gmatrix)]
            else:
                winner = list(orde[i])
            while len(winner) < number_songs:
                if chosen is None:
                    luck = gmatrix / (gpower[winner, :].mean(axis=0) + 0.000001)
                else:
                    #print(gmatrix[chosen].shape, gpower[winner, chosen].shape)
                    luck = gmatrix[chosen] / (gpower[winner, :][:, chosen].mean(axis=0) + 0.000001)
                if np.any(np.isinf(luck)):
                    win = np.argmax(gmatrix[np.isinf(luck)])
                    idx = np.where(np.isinf(luck))[0][win]
                else:
                    idx = int(np.argmax(luck))
                idx += chosen[0] if chosen is not None else 0
                winner.append(idx)
            winners.append(winner)
        return np.array(winners)
    @staticmethod
    def ORDER(dany=None, wyniks=None, test=None, SIGMA=4, matrix= None, distance=None, number_songs = 100, similar = None, periods = 1, uniq = 5, start = 0, orde = None, chosen = None):
        if matrix is None:
            #matrix = Correle.Prediction_Belonging(dany, wyniks, test = test, norm=True)
            matrix = Correle.Prediction_naSterydach(dany, wyniks, norm=True, test=test)
        #matrix = matrix[start:start+periods]
            #matrix = np.vstack(matrix[-1][-320:, :])
        matrix = np.vstack(matrix)
        #print(matrix.shape)
        if distance is None:
            distance = Correle.distance_matrix(matrix=matrix)
        uniq = [121,15,33,76,103,21,17,50,120,124]
        uniq = [114,46,18,41,82,20,6,121,120,124]
        matrix = matrix[:, -5:]
        #matrix = matrix[:, 4::5][:, start:start+periods]
        #print(matrix.shape)
        #matrix = -np.log(1/matrix - 1)
        #matrix = np.mean(matrix, axis=1, keepdims=True)
        #matrix = 1/(1+np.exp(-matrix))
        
        #print(matrix.shape)
        #print("HREE")
        order = Correle.order(matrix=matrix, distance=distance, SIGMA=SIGMA, number_songs=number_songs, orde=orde, chosen=chosen)
        #input()
        return order
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
        #matrix = (matrix - matrix.mean(axis=0))/matrix.std(axis=0)
        #matrix = 1 / (1 + np.exp(-matrix))
        """TRY TO FIND THE SIGMA male"""
        #distance = (matrix[:, None, :] - matrix[None, :, :]) ** 2 @ weight
        #distance = distance ** 0.5
        distance = Correle.distance_matrix(matrix=matrix)
        #distance = (distance - distance.mean())/distance.std()
        SIGMA = np.mean(np.mean(distance ** 2, axis=0)**0.5)/SIGMA
        #print(SIGMA)
        winners = []
        #if similar is None:
        #    uniq = 5
        #else:
        #    uniq = len(similar)
        uniq = [51,32,52,18,0,54,99,71,116,119]
        #power = np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0) / np.mean(np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0))
        power = np.exp(-distance**2/SIGMA**2)
        for i in range(matrix.shape[1]):
            #print(i)
            if similar is None:
                gmatrix = np.zeros(matrix[:, i].shape)
                for p in range(periods):
                    gmatrix += -np.log(1/matrix[:, i]-1)
                gmatrix = gmatrix / (periods ** 0.5)
                gmatrix = 1/(1+np.exp(-gmatrix))
            else:
                cons = distance[:, similar[i]]
                cons = (cons - cons.mean())/cons.std()
                cons = 1/(1+np.exp(cons))
                gmatrix = cons
            gpower = power * gmatrix[:, None]
            winner = [np.argmax(gmatrix)]
            while len(winner) < number_songs:
                luck = gmatrix / (gpower[winner, :].mean(axis=0) + 0.000001)
                if np.any(np.isinf(luck)):
                    win = np.argmax(gmatrix[np.isinf(luck)])
                    #print(np.isinf(luck).sum())
                    idx = np.where(np.isinf(luck))[0][win]
                else:
                    idx = np.argmax(luck)
                #print(gmatrix[idx])
                
                winner.append(idx)
                #Correle.Show_theThing(distance[winner, :][:, [idx]])
            Correle.Show_theThing(np.hstack((-np.log(1/gmatrix[winner][:, None]-1), np.array(winner)[:, None])))
            winners.append(winner)
            #comp = np.hstack((matrix[np.argsort(distance[-i-46-1]), -5:],np.sort(distance[-i-46-1])[:, None]))[:number_songs, :]
            #Correle.Show_theThing(comp)
            #winners.append(np.argsort(distance[-i-46-1])[:number_songs])
            #print(81 - i)
            #print()
            Correle.Show_theThing(matrix[np.argsort(1-gmatrix)[:10], -5:])
            #print(np.argsort(1-gmatrix))
            #input()
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
        power = np.sum(np.exp(-distance ** 2 / SIGMA ** 2), axis=0)
        #print(winners[-1])
        #Correle.Show_theThing(distance[:, -5:])
        #print(SIGMA, np.sum(power)/matrix.shape[0]-1,power.shape)
        return winners
    @staticmethod
    def Correlation_betweenSession(dany, wyniks, test=None):
        if test is None:
            test = dany[:]
        matrix = Correle.Prediction_naSterydach(dany, wyniks, test=test, norm=True)
        Correle.Show_theThing(matrix[-1][:,-5:])
        #pi_c = np.array([matri.shape[0] for matri in matrix])/np.sum(matri.shape[0] for matri in matrix)
        print("OUR SHIT", matrix[-1].shape)
        print("OUR SHIT", matrix[-2].shape)
        print("OUR SHIT", matrix[-3].shape)
        input()
        matrix = Correle.Prediction_naSterydach(dany, wyniks, test=test, norm=True)
        [print(matri.shape) for matri in matrix]
        matrix = np.array([matri.mean(axis=0) for matri in matrix])
        Correle.Show_theThing(matrix[:, [55,17,57,0,101,78,51,127,126,129]])
        print(matrix.shape, 'recent')
        #Correle.Show_theThing(matrix.std(axis=0, keepdims=True))
        #print(matrix.std(axis=0, keepdims=True).sum())
        #D = pi_c[:, None] * pi_c[None, :] * Correle.distance_matrix(matrix=matrix)
        #print(np.sum(D))
        input()
        #matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        dmatrix = Correle.distance_matrix(matrix=matrix)
        Correle.Show_theThing(dmatrix)
        #Correle.Show_theThing(1/(1+np.exp(-(dmatrix-dmatrix.mean())/dmatrix.std())))
        print(dmatrix.shape, 'distance')
        input()
        Correle.Show_theThing(matrix[:, 4::5])
        print(matrix.shape, 'likes')
        input()
        Correle.Show_theThing(matrix[:, -5:])
        print(matrix.shape, 'recent')
        input()
        Correle.Show_theThing(matrix[:, :])
        print(matrix.shape, 'all shit')
        input()
        matrix = Correle.Correaltion(matrix=matrix[:, 4::5], axis=1)
        Correle.Show_theThing(matrix)
        #matrix = np.argsort(matrix)
        #Correle.Show_theThing(np.where(matrix > 74, matrix, 0))
        print(matrix.shape, 'correlation')
        input()
    @staticmethod
    def lowerdimension(dany = None, wyniks = None, distance = None, test = None):
        matrix = Correle.Prediction_naSterydach(dany, wyniks, norm=True, test=test)
        matrix = np.array([matri.mean(axis=0) for matri in matrix])
        matrix = np.vstack(matrix)
        #matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        #matrix = matrix.T
        #matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        #matrix = 1 / (1 + np.exp(-matrix))
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
        pos = (np.isreal(values)) & (values > 0.0000001)
        values = values[pos].real
        vectors = vectors[:, pos].real

        window = 100

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
        Correle.Show_theThing(x)
        print(x.shape)
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
    @staticmethod
    def K_mean(matrix, k_mean = 10):
        distance = Correle.distance_matrix(matrix=matrix)
        seed = list(range(k_mean))
        choosen = np.argmin(distance[:k_mean, :], axis=0)
        power = [np.where(choosen==i)[0] for i in range(k_mean)]
        print([len(powe) for powe in power])
        for i in range(10):
            common = np.vstack([np.mean(matrix[powe, :], axis=0) for powe in power])
            Correle.Show_theThing(common[:, -5:])
            common = np.vstack((common, matrix))
            k_distance = Correle.distance_matrix(matrix=common)
            #seed = np.argmin(k_distance[k_mean:, :k_mean], axis=0)
            choosen = np.argmin(k_distance[:k_mean, k_mean:], axis=0)
            #choosen = np.argmin(distance[seed, :], axis=0)
            power = [np.where(choosen==j)[0] for j in range(len(seed))]
            print([len(powe) for powe in power])
        return power
    
    @staticmethod
    def PCA(matrix):
        dane_centre = matrix - matrix.mean(axis=0)
        covariance = (dane_centre.T @ dane_centre) / dane_centre.shape[0]
        values, vectors = np.linalg.eig(covariance)
        idx = np.argsort(values)[::-1]
        values = values[idx]
        vectors = vectors[:, idx].real
        pos = (np.isreal(values)) & (values > 0.0000001)
        vectors = vectors[:, pos]
        TTT = dane_centre @ vectors
        return TTT
    
    @staticmethod
    def GMM(matrix, k_mean = 10):
        epsilon = 0.00001
        n_agent = matrix.shape[0]

        compressed = Correle.PCA(matrix=matrix)
        idx = np.random.choice(n_agent, k_mean, replace=False)
        idx = list(range(k_mean))
        means_c = compressed[:k_mean, :]

        diff = compressed[:, None, :] - means_c[None, :, :]
        print(diff.shape)
        stds_c = diff.transpose(1,2,0) @ diff.transpose(1,0,2) / n_agent
        print(stds_c.shape)
        stds_c += np.eye(stds_c.shape[1]) * epsilon
        pi_c = np.ones(k_mean) / k_mean
        for i in range(10):
            inv_stds_c = np.linalg.inv(stds_c)
            norm_const = ((2 * np.pi) ** (compressed.shape[1]/2)) * np.sqrt(np.linalg.det(stds_c))
            #print(np.linalg.det(stds_c))
            #print(np.linalg.det(stds_c * (10000)))
            exponent = -0.5 * diff[:,:,None,:] @ inv_stds_c[None,:,:,:] @ diff[:,:,:,None]
            exponent = exponent.squeeze(-1).squeeze(-1)
            exponent = np.clip(exponent, -300,300)
            prob = np.exp(exponent) / norm_const
            
            lihoood = prob * pi_c
            lihood = lihoood / lihoood.sum(axis=1)[:, None]
            
            n_dudes = lihood.sum(axis=0) + epsilon
            pi_c = n_dudes / n_agent
            print(pi_c)

            #UPDATE
            means_c = lihood.T @ compressed / n_dudes[:,None]
            if i == 9:
                dist = Correle.distance_matrix(matrix=means_c)
                D = np.sum(pi_c[:, None] * pi_c[None, :] * dist)
                print(D)
                input()
            #Correle.Show_theThing(np.where(lihood>0.001, lihood,0))
            #print(pi_c)
            #input()
            diff = compressed[:,None,:] - means_c[None,:,:]
            stds_c = diff.transpose(1,2,0) * lihood.T[:,None,:] @ diff.transpose(1,0,2) / n_dudes[:,None,None]
            stds_c = 0.5 * (stds_c + stds_c.transpose(0,2,1)) + np.eye(compressed.shape[1]) * epsilon
        print(np.where(lihood[:, 0] > 0.1, lihood[:, 0], 0))
        lihod = lihood /lihood.sum(axis=0)
        steps = (1 + 5**0.5)/2-1
        order = []
        for j in range(k_mean):
            order.append([])
            total = 0
            while len(order[-1]) < 200:
                for i in range(n_agent):
                    total += lihod[i,j]
                    if total > steps:
                        total -= steps
                        order[-1].append(i)
        win = np.where(lihood > 0.5)
        winners = np.vstack([np.mean(matrix[orde, :], axis=0) for orde in win])
        Correle.Show_theThing(winners[:, -5:])
        print()
        Correle.Show_theThing(winners.std(axis=0, keepdims=True))
        print(winners.std(axis=0, keepdims=True).sum())
        input()
        winners = (winners - winners.mean(axis=0))/winners.std(axis=0)
        Correle.Show_theThing(winners[:, -5:])
        return order
        #return np.argsort(-lihoood, axis=0)[:200, :].T
    #ONLY USING SINGLE DATASET
    @staticmethod
    def Expand_system_experiment(raw, wyn):
        raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        wyn = (wyn - wyn.mean(axis=0)) / wyn.std(axis=0)
        raw = 1 / (1 + np.exp(-raw))
        wyn = 1 / (1 + np.exp(-wyn))
        token_id = []
        for i in range(raw.shape[1] - 1):
            for j in range(raw.shape[1] - 1 - i):
                for k in (-1, 1):
                    for l in (-1, 1):
                        newone = np.zeros(raw.shape[1])
                        newone[i] = k
                        newone[j+i+1] = l
                        token_id.append(newone)
        goal_token = []
        for i in range(wyn.shape[1]):
            best_error = []
            for token in token_id:
                pred_test = raw * token
                win_mask = (pred_test != 0)
                winn = pred_test * win_mask  
                winn = np.where(winn > 0, winn, winn + 1)
                winn = np.min(winn, axis=1)
                error = np.mean((np.log(1/winn-1) - np.log(1/wyn[:, i]-1)) ** 2) ** 0.5
                best_error.append(error)
            goal_token.append(token_id[np.argmin(best_error)])
            print(goal_token[-1], np.min(best_error))
        #nodes_win = np.array([raw * token for token in token_id])
        nodes_win = raw[: , None, :] * np.array(token_id)[None, :, :]
        nodes_win = np.where(nodes_win > 0, nodes_win, nodes_win + 1)
        nodes_win = np.min(nodes_win, axis=2)
        wyn_nodes = raw[: , None, :] * np.array(goal_token)[None, :, :]
        wyn_nodes = np.where(wyn_nodes > 0, wyn_nodes, wyn_nodes + 1)
        wyn_nodes = np.min(wyn_nodes, axis=2)
        hidden = np.arange(nodes_win.shape[1])
        print(np.array(goal_token).shape, wyn_nodes.shape)
        print(nodes_win.shape)
        while True:
            true_token = []
            for i in range(nodes_win.shape[1] - 1):
                for j in range(nodes_win.shape[1] - 1 - i):
                    for k in (-1, 1):
                        for l in (-1, 1):
                            newone = np.zeros(nodes_win.shape[1])
                            newone[i] = k
                            newone[j+i+1] = l
                            true_token.append(newone)
            print(len(true_token))
            true_goal = []
            for i in range(wyn.shape[1]):
                best_error = []
                for token in true_token:
                    pred_test = nodes_win * token
                    winn = np.where(pred_test > 0, pred_test, pred_test + 1)
                    winn = np.min(winn, axis=1)
                    error = np.mean((np.log(1/winn-1) - np.log(1/wyn[:, i]-1)) ** 2) ** 0.5
                    best_error.append(error)
                true_goal.append(true_token[np.argmin(best_error)])
                print(true_goal[-1], np.min(best_error))
            anihila = []
            for i in range(len(hidden)):
                if np.sum(np.abs(np.array(true_goal)[:, i])) < 1:
                    anihila.append(i)
            nodes_win = np.delete(nodes_win, anihila, axis=1)
            hidden = np.delete(hidden, anihila)
            print(hidden, anihila)
            if len(anihila) < 1:
                break
        print("DONE")
        input()
        print(hidden)
        for hide in hidden:
            print(token_id[hide])
        return True

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