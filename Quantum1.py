import numpy as np

class NeuralNet(object):

    def __init__(self, N, M):
        self.weights = []
        for i in range(N):
            tempL = []
            for j in range(M):
                tempL.append(np.random.normal(N,N-1))
            self.weights.append(tempL)
        self.weights = np.array(self.weights)