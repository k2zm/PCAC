import numpy as np

class Clustering:
    def __init__(self, tau):
        self.tau = tau
        self.tau_1 = 1/tau
        self.embwa_list = []
        self.anom_list = []

    def weight_average(self, embedding, distances):
        distances = distances[:, 0]
        alpha = np.exp(self.tau_1 * distances)
        alpha = alpha / np.sum(alpha)

        embwa = (alpha * embedding).sum(axis=(0))

        return embwa
    
    def append_embwa(self, embwa, anomname):
        self.embwa_list.append(embwa)
        self.anom_list.append(anomname)

