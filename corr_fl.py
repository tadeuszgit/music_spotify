import numpy as np
from data_import import data_extraction


class Corrl:
    def __init__(self):
        impo = data_extraction()
        self.raws_origin, self.wyniks_origin = impo.Open_Multiple()
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
        m = np.linalg.solve(A, b).squeeze(-1).T
        if norm:
            m[0, :] -= self.out.mean(axis=0)
            m /= self.out.std(axis=0)
        return m
    def Coefficient_Regulation(self, energy = None, LAMBDA = None, dim = 2, norm = False):
        self.inps, self.outs = self.raws_origin, self.wyniks_origin
        if LAMBDA is not None:
            if LAMBDA == 0:
                goal_inp = np.hstack((0, np.std(np.vstack(self.inps), axis=0)))
                goal_out = np.hstack([np.std(out, axis=0) for out in self.outs])
                if energy is not None:
                    for i in range(1000):
                        m = self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=False)
                        TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
                        TRUE_ENERGY_PERY = np.sum(TRUE_ENERGY, axis=0, keepdims=True) / energy
                        TRUE_ENERGY = np.mean(np.sum(TRUE_ENERGY, keepdims=True, axis=0), keepdims=True) / energy
                        stay = np.log(TRUE_ENERGY) / TRUE_ENERGY / energy + np.log(TRUE_ENERGY_PERY) / TRUE_ENERGY_PERY / energy
                        stay = stay * 2 * dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim) * np.median(np.abs(m), keepdims=True, axis=1) ** (dim - 2)
                        stay = np.where(np.abs(stay) > 10**100, 10**100*np.sign(stay), stay)
                        LAMBDA = stay * 0.01 + LAMBDA * 0.99
                else:
                    energy = 1
                    energi = 1
                    for j in range(10):
                        for i in range(1000):
                            m = self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=False)
                            TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
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
                        #print(np.mean(LAMBDA, keepdims=True, axis=1))
                        print(LAMBDA[:, :4])
                        print(energy[0, -5:])
                        print(energi)
        return self.Coefficient_for_all_dataset(LAMBDA=LAMBDA, norm=norm)
    def Coefficient_for_all_dataset(self, LAMBDA = None, norm = False):
        #self.inps, self.outs = self.raws_origin, self.wyniks_origin
        m = []
        k = 0
        for inp, out in zip(self.inps, self.outs):
            self.inp, self.out = inp/np.std(inp, axis=0), out
            if type(LAMBDA) == int or type(LAMBDA) == float:
                m.append(self.Coefficient(LAMBDA=LAMBDA, norm=norm))
            else:
                m.append(self.Coefficient(LAMBDA=LAMBDA[:, k:k+self.out.shape[1]], norm=norm))
                k += self.out.shape[1]
        return np.hstack(m)