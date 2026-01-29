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
    def Coefficient_Regulation(self, energy, LAMBDA, dim = 2, norm = False):
        if LAMBDA is not None:
            if LAMBDA == 0:
                goal_inp = np.hstack((0, np.std(self.inp, axis=0)))
                goal_out = np.std(self.out, axis=0)
                if energy is not None:
                    for i in range(1000):
                        m = self.Coefficient(LAMBDA=LAMBDA, norm=False)
                        TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
                        TRUE_ENERGY = np.sum(TRUE_ENERGY, axis=0, keepdims=True) / energy
                        stay = 2 * np.log(TRUE_ENERGY) * dim / TRUE_ENERGY * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim * energy) * np.abs(m) ** (dim - 2)
                        stay = np.where(np.abs(stay) > 10**100, 10**100*np.sign(stay), stay)
                        LAMBDA = stay * 0.1 + LAMBDA * 0.9
                else:
                    energy = 1
                    for j in range(10):
                        for i in range(1000):
                            m = self.Coefficient(LAMBDA=LAMBDA, norm=False)
                            TRUE_ENERGY = np.abs(m) ** dim * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim)
                            TRUE_ENERGY = np.sum(TRUE_ENERGY, axis=0, keepdims=True) / energy
                            stay = 2 * np.log(TRUE_ENERGY) * dim / TRUE_ENERGY * goal_inp[:, None] ** dim / (goal_out[None, :] ** dim * energy) * np.abs(m) ** (dim - 2)
                            stay = np.where(np.abs(stay) > 10**100, 10**100*np.sign(stay), stay)
                            LAMBDA = stay * 0.1 + LAMBDA * 0.9
                        energy = TRUE_ENERGY * 0.5 + energy * 0.5
        return self.Coefficient(LAMBDA=LAMBDA, norm=norm)
    def Coefficient_for_all_dataset(self, energy = None, LAMBDA = None, dim = 2, norm = False):
        self.inps, self.outs = self.raws_origin, self.wyniks_origin
        m = []
        for inp, out in zip(self.inps, self.outs):
            self.inp, self.out = inp, out
            m.append(self.Coefficient_Regulation(energy=energy, LAMBDA=LAMBDA, dim=dim, norm=norm))
        return np.hstack(m)