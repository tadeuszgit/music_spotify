import numpy as np
from data_import import data_extraction


class Corrl:
    def __init__(self):
        impo = data_extraction()
        self.raws_origin, self.wyniks_origin = impo.Open_Multiple()
        self.tokens = []
        self.datas = [self.raws_origin, self.wyniks_origin]
    def Predict(self, M, norm = False):
        pred = []
        for inp in self.inps:
            ones = np.ones([inp.shape[0], 1])
            ipn = np.hstack([ones, inp])
            pre = ipn @ M
            if norm:
                pre = np.clip(pre, -300, 300)
                pre = 1 / (1 + np.exp(-pre))
            pred.append(pre)
        return np.vstack(pred)
    def Coefficient(self, LAMBDA, norm = False):
        #self.inp, self.out
        inp = np.repeat(self.inp[np.newaxis, :, :], self.out.shape[1], axis=0)
        out = self.out.T[:, :, np.newaxis]
        ones = np.ones((inp.shape[0], inp.shape[1], 1))
        matrix = np.concatenate([ones, inp, out], axis=2)
        mat = matrix[:, :, :-1].transpose(0,2,1) @ matrix
        A = mat[:, :, :-1]
        b = mat[:, :, -1:]
        if LAMBDA is not None:
            if type(LAMBDA) == int or type(LAMBDA) == float:
                A += LAMBDA * np.repeat(np.eye(A.shape[1])[np.newaxis, :, :], A.shape[0], axis=0) * matrix[:, :, :-1].std(axis=1, keepdims=True) ** 2
            else:
                LAMBDA = np.eye(A.shape[1])[None] * LAMBDA.T[:,:,None]
                A += LAMBDA
        #m = np.array([np.linalg.lstsq(A[i], b[i])[0] for i in range(A.shape[0])]).squeeze(-1).T
        m = np.array([self.safe_lin(A[i], b[i]) for i in range(A.shape[0])]).squeeze(-1).T
        #m = np.linalg.solve(A, b).squeeze(-1).T
        if norm:
            m[0, :] -= self.out.mean(axis=0)
            stds = self.out.std(axis=0)
            stds = np.where(stds > 0.001, stds, 0.001)
            m /= stds
        return m
    @staticmethod
    def safe_lin(A, b):
        try:
            #print("GOOOD")
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            #print("AVOIDING ERROR")
            return np.linalg.lstsq(A, b)[0]
    def Coefficient_Regulation(self, energy = None, LAMBDA = None, dim = 2, norm = False):
        #self.inps, self.outs = self.raws_origin, self.wyniks_origin
        if LAMBDA is not None:
            if LAMBDA == 0:
                goal_inp = np.hstack((0, np.std(np.vstack(self.inps), axis=0)))
                goal_out = np.hstack([np.std(out, axis=0) for out in self.outs])
                goal_out = np.where(goal_out == 0, 0.0001, goal_out)
                #print(goal_out)
                if energy is not None:
                    for i in range(100):
                        m = self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=False)
                        TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
                        TRUE_ENERGY_PERY = np.sum(TRUE_ENERGY, axis=0, keepdims=True) / energy
                        TRUE_ENERGY = np.mean(np.sum(TRUE_ENERGY, keepdims=True, axis=0), keepdims=True) / energy
                        stay = np.log(TRUE_ENERGY) / TRUE_ENERGY / energy + np.log(TRUE_ENERGY_PERY) / TRUE_ENERGY_PERY / energy
                        stay = stay * 2 * dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim) * np.median(np.abs(m), keepdims=True, axis=1) ** (dim - 2)
                        stay = np.where(np.abs(stay) > 10**100, 10**100*np.sign(stay), stay)
                        LAMBDA = stay * 0.1 + LAMBDA * 0.9
                        print(np.mean(TRUE_ENERGY_PERY), i)
                else:
                    energy = 1
                    energi = 1
                    for j in range(10):
                        for i in range(100):
                            m = self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=False)
                            TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
                            #TRUE_ENERGY[is.]
                            #print(TRUE_ENERGY[0, :], TRUE_ENERGY.shape)
                            #input()
                            TRUE_ENERGY_PERY = np.sum(TRUE_ENERGY, axis=0, keepdims=True) / energy
                            #TRUE_ENERGY = np.where(TRUE_ENERGY == 0, 1, TRUE_ENERGY)
                            TRUE_ENERGY = np.mean(np.sum(TRUE_ENERGY, keepdims=True, axis=0), keepdims=True) / energi
                            stay = np.log(TRUE_ENERGY) / TRUE_ENERGY / energi + np.log(TRUE_ENERGY_PERY) / TRUE_ENERGY_PERY / energy
                            #stay = np.log(TRUE_ENERGY_PERY) / TRUE_ENERGY_PERY
                            #np.mean(np.abs(m), keepdims=True, axis=1)
                            stay = stay * 2 * dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim) * np.median(np.abs(m), keepdims=True, axis=1) ** (dim - 2)
                            #stay = 2 * np.log(TRUE_ENERGY) * dim / TRUE_ENERGY * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim * energy) * np.abs(m) ** (dim - 2)
                            stay = np.where(np.abs(stay) > 10**100, 10**100*np.sign(stay), stay)
                            LAMBDA = stay * 0.1 + LAMBDA * 0.9
                        #energy = energy * 0.5 * (1 + TRUE_ENERGY_PERY)
                        #energi = energi * 0.5 + TRUE_ENERGY*energi*0.5
                        energy = energy * TRUE_ENERGY_PERY
                        energi = energi * TRUE_ENERGY
                        print(energi, j)
                        #print(energi)
                        #print(np.mean(LAMBDA, keepdims=True, axis=1))
        return self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=norm)
    def Coefficient_for_all_dataset(self, LAMBDA = None, norm = False):
        #self.inps, self.outs = self.raws_origin, self.wyniks_origin
        m = []
        k = 0
        for inp, out in zip(self.inps, self.outs):
            self.inp, self.out = inp, out
            if type(LAMBDA) == int or type(LAMBDA) == float:
                m.append(self.Coefficient(LAMBDA=LAMBDA, norm=norm))
            else:
                m.append(self.Coefficient(LAMBDA=LAMBDA[:, k:k+self.out.shape[1]], norm=norm))
                k += self.out.shape[1]
        return np.hstack(m)
    def Get_tokens(self):
        self.tokens, self.datas
        #self.tokens = []
        for data in self.datas[-2:-1]:
            fea_n = data[0].shape[1]
            token = []
            for i in range(fea_n - 1):
                for j in range(fea_n - 1 - i):
                    for k in (-1, 1):
                        for l in (-1, 1):
                            newone = np.zeros(fea_n)
                            newone[i] = k
                            newone[i+j+1] = l
                            token.append(newone)
            print("TOKEN!!!!")
            print(len(token))
            self.tokens.append(token)
    def Get_dane_from_token(self):
        self.tokens, self.datas
        #self.datas = [self.datas[0], self.datas[-1]]
        datas = self.datas[:-1]
        for token in self.tokens[-1:]:
            print(np.vstack(datas[-1])[:, None, :].shape, np.array(token)[None, :, :].shape)
            input()
            inp = np.vstack(datas[-1])[:, None, :] * np.array(token)[None, :, :]
            
            inp = np.where(inp > 0, inp, inp+1)
            inp = np.min(inp, axis=2)
            self.inps, self.outs = [np.vstack(datas[-1])], [inp]
            print("MMMMM")
            m = self.Coefficient_Regulation(LAMBDA = 0, norm=True, energy=1)
            print(np.sum(np.abs(m) < 0.01, axis=1))
            print("PREDICTION")
            inp = self.Predict(m, norm = True)
            datas.append([inp])
        datas.append(self.datas[-1])
        self.datas = []
        for data in datas:
            self.datas.append(data)
    def Select_token(self):
        for i in range(len(self.tokens)):
            self.inps, self.outs = self.datas[-i-2], self.datas[-i-1]
            if i == 0:
                inps = []
                k = 0
                for out in self.outs:
                    inps.append(self.inps[0][k:k+out.shape[0], :])
                self.inps = inps
            print("SELECTION")
            winn = np.arange(len(self.tokens[i]))
            while len(winn) > 10:
                m = self.Coefficient_Regulation(LAMBDA = 0, norm = True, dim=1, energy=1)
                kand = np.sum(np.abs(m) < 0.01, axis=1)
                print(kand)
                bord = min(np.max(kand),3)
                win = kand[1:] < bord
                winn = winn[win]
                self.inps = [inp[:, win] for inp in self.inps]
                print(winn)
                print(m.shape, len(winn))
            self.datas[-i-2] = self.inps
            [print(len(np.vstack(dan))) for dan in self.datas]
            print("DONE", i, len(self.tokens))
            input()
