import numpy as np

N = 5
M = 5

#wavefunction is a vector of coefficients
def Hamiltonian(wavefunction):
    #The Ising hamiltonian in 1D
    H = 0
    for i in range(N-1):
        H += (wavefunction[i])*(wavefunction[i+1])
    return H

class NeuralNet(object):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.weights = []
        for i in range(N):
            tempL = []
            for j in range(M):
                tempL.append(np.random.normal(N,N-1)*np.sqrt(2/(N-1)))
            self.weights.append(tempL)
        self.weights = np.array(self.weights)

    #a is a 1xN vector bias
    #b is a 1xM vector bias
    #hidden is a 1xM vector
    #wavefunction is a 1xN vector
    #weights is a NxM matrix
    def Frbm(a, b, hidden, wavefunction, weights):
        E = 0
        for i in range(N):
            E -= a[i]*wavevfuncion[i]
        for i in range(M):
            E -= b[i]*hidden[i]
        for i in range(N):
            v = wavefunction[i]
            for j in range(M):
                E -= weights[i][j]*v*hidden[j]
        return E

