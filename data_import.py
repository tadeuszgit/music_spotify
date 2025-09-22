import os
import numpy as np

class data_extraction:
    def __init__(self, raw='data\\raw\\', dane='data\\result\\'):
        self.rawnes = raw
        self.result = dane
        self.file_raw = os.listdir(raw)
        self.file_res = os.listdir(dane)
    
    def Open_Input(self, filename, both = True, two = True):
        data = open(self.rawnes + filename, 'r')
        raw = []
        for row in data:
            dane = row.split()
            for i in range(len(dane)):
                dane[i] = float(dane[i].replace(',', '.'))
            raw.append(dane)
        if not both:
            return raw
        data = open(self.result + filename, 'r')
        wynik = []
        for row in data:
            dane = row.split()
            for i in range(len(dane)):
                dane[i] = float(dane[i].replace(',', '.'))
            wynik.append(dane)
        if two:
            return raw, wynik
        for i in range(len(raw)):
            for wyn in wynik[i]:
                raw[i].append(wyn)
        return raw
    
    def Open_Multiple(self, all_com = True, both = True, two = True):
        if all_com:
            filenames = self.file_res
        else:
            filenames = self.file_raw
        if both:
            if two:
                raws = []
                wyniks = []
                for filename in filenames:
                    raw, wynik = self.Open_Input(filename=filename)
                    raw = np.array(raw)
                    wynik = np.array(wynik)
                    raws.append(raw)
                    wyniks.append(wynik)
                return raws, wyniks
            raws = []
            for filename in filenames:
                raw = self.Open_Input(filename=filename, two=False)
                raw = np.array(raw)
                raws.append(raw)
            return raws
        raws = []
        for filename in filenames:
            raw = self.Open_Input(filename=filename, both=False)
            raw = np.array(raw)
            raws.append(raw)
        return raws
