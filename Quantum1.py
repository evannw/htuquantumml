import numpy as np


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
#returns a vector that's a modified wavefunction
def Hsys(N, wavefunction):
    Hsysn = np.zeros(2**N)
    for i,coeff in enumerate(wavefunction): 
        Hsysn[i] = (Hcoeff(N, state_from_iteration(N, i))*coeff)
    return Hsysn
    
#Returns a Hamiltonian diagonal matrix of size 2^nx2^n using Ising model
def Hamiltonian(N):
    Hsysn = np.zeros((2**N,2**N))
    for i in range(N): 
        Hsysn[i][i] = Hcoeff(N, state_from_iteration(N, i))
    return Hsysn

class NeuralNet(object):

    def __init__(self, N, M, D):
        self.N = N
        self.M = M
        self.a = np.random.random(N)
        self.b = np.random.random(M)
        #D is learning rate (used in update)
        self.D = D
        #Computes a generic hamiltonian matrix for size N
        self.hamiltonian = Hamiltonian(N)
        self.weights = []
        self.wavefunction = []
        for i in range(N):
            tempL = []
            for j in range(M):
                # tempL.append(np.random.normal(N,N-1)*np.sqrt(2/(N-1)))
                tempL.append(np.random.random())
            self.weights.append(tempL)
        self.weights = np.array(self.weights)
        self.updateWF()

    # #function to compute current wave function
    # def psi(self):
    #     wf = np.zeros(2**self.N)
    #     for i in range(2**self.N):#iterate through spin states
    #         spin_state = state_from_iteration(self.N, i)
    #         # for h in range(2**self.M):#iterate through h states
    #             # h_state = self.state_from_iteration(h)
    #             # next we compute the coefficient for this state
    #             # wf[i] += np.exp(self.sum_states(spin_state, h_state))
    #         x = np.exp(np.dot(spin_state,self.a))
    #         y = np.prod(np.matmul(np.transpose(self.weights), spin_state) + self.b)
    #         wf[i] = x*y
    #     self.wavefunction = wf
    #     return wf
    
    #helper function to compute the sum in the exponential
    def sum_states(self,spin_state, h_state):
        summation = 0
        for i in range(len(spin_state)):
            for j in range(len(h_state)):
                summation += self.weights[i,j]*spin_state[i]*h_state[j]
        return summation
    
    def updateWF(self):
        wf = []
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            theta = 1 #product term in the wave function
            for j in range(self.M):
                theta *= 2*np.cosh(np.dot(self.weights[:,j], spin_state) + self.b[j])
            x = np.exp(np.dot(spin_state,self.a))
            theta *= x
            wf.append(theta)
        self.wavefunction = wf
        return wf

    def grad(self):
    def grad_psi(self):
        del_psi = []
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            # param_index = 0
            self.updateWF()
            
            db = np.array([self.wavefunction[i]*(np.tanh(np.dot(self.weights[:,j], spin_state) + self.b[j])) for j in range(self.M)])

            dw = np.array([[db[m]*spin_state[n] for m in range(self.M)] for n in range(self.N)])

            da = np.array([spin_state[k]*self.wavefunction[i] for k in range(self.N)])
            
            dpi = np.append(np.append(da,db),dw)
            del_psi = np.append(del_psi, dpi)
    def grad_e(self):
        del_psi = grad_psi()
        bra_wf = np.transpose(self.wavefunction)
        right = np.matmul(self.hamiltonian,del_psi)
        return 2*(np.real(np.matmul(bra_wf,right)) + self.EnergyExpectation()*np.real(np.matmul(bra_wf,del_psi)))/norm(self.wavefunction)

    #a is a 1xN vector bias
    #b is a 1xM vector bias
    #hidden is a 1xM vector
    #wavefunction is a 1xN vector
    #weights is a NxM matrix
    def Frbm(self, a, b, hidden, wavefunction, weights):
        E = 0
        for i in range(self.N):
            E -= a[i]*wavefunction[i]
        for i in range(self.M):
            E -= b[i]*hidden[i]
        for i in range(self.N):
            v = wavefunction[i]
            for j in range(self.M):
                E -= weights[i][j]*v*hidden[j]
        return E

    def EnergyExpectation(self):
        Hsysn = Hsys(self.N, self.wavefunction)
        return np.dot(self.wavefunction, Hsysn)/norm(self.wavefunction)

    def UpdateOnce(self):
        gradient = self.grad_e()
        for i in range(self.N):
            self.a[i] -= self.D*gradient[i]
        for i in range(self.N, self.N+self.M):
            self.b[i-self.N] -= self.D*gradient[i]
        for i in range(self.N):
            for j in range(self.M):
                self.weights[i][j] -= self.D*gradient[self.N+self.M+self.M*i+j]

def Train():
    test = NeuralNet(2,2,1)
    test.psi()
    for i in range(3):
        print(test.wavefunction)
        print(test.EnergyExpectation())
        test.UpdateOnce()

Train()
# test1 = NeuralNet(2, 2, 0, 0, 1)
# test1.psi()
# print(test1.EnergyExpectation())
