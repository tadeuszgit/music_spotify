import numpy as np
from data_import import data_extraction


class Corrl:
    def __init__(self):
        impo = data_extraction()
        self.raws_origin, self.wyniks_origin = impo()
    def Coefficient(self, LAMBDA, norm = False):
        #self.inp, self.out
        inp = np.repeat(self.inp[np.newaxis, :, :], self.out.shape[1], axis=0)
        out = self.out.T[:, :, np.newaxis]
        ones = np.ones((inp.shape[0], inp.shape[1], 1))
        matrix = np.concatenate([ones, inp, out], axis=2)

        mat = matrix[:, :, :-1].transpose(0,2,1) @ matrix
        A = mat[:, :, :-1]
        b = mat[:, :, -1:]
        if LAMBDA:
            if type(LAMBDA) == int or type(LAMBDA) == float:
                A += LAMBDA * np.repeat(np.eye(A.shape[1])[np.newaxis, :, :], A.shape[0], axis=0) * matrix[:, :, :-1].std(axis=1, keepdims=True) ** 2
            else:
                A += LAMBDA
        
        m = np.linalg.solve(A, b).squeeze(-1).T

        if norm:
            m[0, :] -= self.out.mean(axis=0)
            m /= self.out.std(axis=0)
        
        return m