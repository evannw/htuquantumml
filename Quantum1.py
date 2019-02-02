import numpy as np

N = 5
M = 5


def norm(wavefunction):
    return np.sqrt(np.sum([x**2 for x in wavefunction]))

#to iterate through spin states, we just count up to 2**n and
#use the binary representation of the count as our state,
#replacing 0's with -1's
def state_from_iteration(N, i):
    i_bin = bin(i)[2:]#bin gives you '0b#'
    i_arr = [int(ch) for ch in i_bin]#now we have an array of 1's and 0's
    i_arr = list(np.zeros(N - len(i_arr))) + i_arr
    for k, spin in enumerate(i_arr):#convert all 0's to -1's
        if spin == 0:
            i_arr[k] = -1
    return i_arr

#state is a vector state of system
def Hcoeff(N, state):
    #The Ising hamiltonian in 1D
    H = 0
    for i in range(N-1):
        H += (state[i])*(state[i+1])
    return H

#wavefunction is a vector of length 2^n
#returns a diagonal matrix of modified state 
def Hsys(N, wavefunction):
    Hsysn = np.zeros(2**N)
    for i,coeff in enumerate(wavefunction): 
        Hsysn[i] = (Hcoeff(N, state_from_iteration(N, i))*coeff)
    return Hsysn
    
class NeuralNet(object):

    def __init__(self, N, M, a, b):
        self.N = N
        self.M = M
        self.a = a*np.ones((N, 1))
        self.b = b*np.ones((M, 1))
        self.weights = []
        self.wavefunction = []
        for i in range(N):
            tempL = []
            for j in range(M):
                tempL.append(np.random.normal(N,N-1)*np.sqrt(2/(N-1)))
            self.weights.append(tempL)
        self.weights = np.array(self.weights)

    #function to compute current wave function
    def psi(self):
        wf = np.zeros(2**self.N)
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            # for h in range(2**self.M):#iterate through h states
                # h_state = self.state_from_iteration(h)
                # next we compute the coefficient for this state
                # wf[i] += np.exp(self.sum_states(spin_state, h_state))
            x = np.exp(np.dot(spin_state,self.a))
            y = np.prod(np.matmul(np.transpose(self.weights), spin_state) + self.b)
            wf[i] = x*y
        self.wavefunction = wf
        return wf
    
    #helper function to compute the sum in the exponential
    def sum_states(self,spin_state, h_state):
        summation = 0
        for i in range(len(spin_state)):
            for j in range(len(h_state)):
                summation += self.weights[i,j]*spin_state[i]*h_state[j]
        return summation

    def grad(self, hamiltonian):
        del_psi = np.zeros((2**self.N, self.weights.size + self.a.size + self.b.size))
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            param_index = 0

            theta = 1#product term in the wave function
            for j in range(M):
                theta *= 2*np.cosh(np.dot(self.weights[:,j], spin_state) + self.b[j])
            for a_ in self.a:#partial with respect to each a
                del_psi[i,param_index] = spin_state[param_index]*np.exp(np.dot(spin_state,self.a))*theta
                param_index += 1

            for [m, n], weight in np.ndenumerate(self.weights):#partial with respect to each weight
                theta0 = theta/2*cosh(np.dot(self.weights[:,n], spin_state) + self.b[n])
                del_psi[param_index] = theta0*2*np.sinh(np.dot(self.weights[:n],spin_state) + self.b[n])*spin_state[m]
                if m == 0:
                    del_psi[param_index + self.weights.size] = theta0*2*np.sinh(np.dot(self.weights[:n],spin_state) + self.b[n])*spin_state[m]
                param_index += 1
        #compute gradient of the energy using product rule
        right = np.matmul(hamiltonian,del_psi)
        return 2*np.real(np.matmul(np.transpose(self.wavefunction),right))

    #a is a 1xN vector bias
    #b is a 1xM vector bias
    #hidden is a 1xM vector
    #wavefunction is a 1xN vector
    #weights is a NxM matrix
    def Frbm(a, b, hidden, wavefunction, weights):
        E = 0
        for i in range(N):
            E -= a[i]*wavefunction[i]
        for i in range(M):
            E -= b[i]*hidden[i]
        for i in range(N):
            v = wavefunction[i]
            for j in range(M):
                E -= weights[i][j]*v*hidden[j]
        return E

    def EnergyExpectation(self):
        Hsysn = Hsys(self.N, self.wavefunction)
        print(Hsysn)
        return np.matmul(self.wavefunction, Hsysn)/norm(self.wavefunction)

    

test1 = NeuralNet(2, 2, 0, 0)
test1.psi()
print(test1.EnergyExpectation())
