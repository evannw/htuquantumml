import numpy as np

sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])

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

def Hamiltonian_heisenberg(N):
    ham = np.zeros((2**N,2**N))
    for i in range(N-1):
        product_terms = [np.identity(2) for i in range(N)]
        #compute x products
        product_terms[i:i+2] = [sx,sx]
        x_prod = np.ones(1)
        for term in product_terms:
            x_prod = np.kron(x_prod,term)
        #compute y products
        product_terms[i:i+2] = [sy,sy]
        y_prod = np.ones(1)
        for term in product_terms:
            y_prod = np.kron(y_prod,term)
        #compute z products
        product_terms[i:i+2] = [sz,sz]
        z_prod = np.ones(1)
        for term in product_terms:
            z_prod = np.kron(z_prod,term)
        #add all the products to the hamiltonian
        ham = ham + x_prod + y_prod + z_prod
    return ham

class NeuralNet(object):

    def __init__(self, N, M, D):
        self.N = N
        self.M = M
        self.a = np.random.random(N)
        self.b = np.random.random(M)
        #D is learning rate (used in update)
        self.D = D
        #Computes a generic hamiltonian matrix for size N
        self.hamiltonian = Hamiltonian_heisenberg(N)
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

    def updateWF(self):
        wf = []
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            theta = 1 #product term in the wave function
            #print("Weights")
            #print(self.weights)
            #print("Spin State")
            #print(spin_state)
            for j in range(self.M):
                theta *= 2*np.cosh(np.dot(self.weights[:,j], spin_state) + self.b[j])
            x = np.exp(np.dot(spin_state,self.a))
            theta *= x
            wf.append(theta)
        self.wavefunction = wf/(norm(wf))
        return self.wavefunction

    def grad_psi(self):
        del_psi = np.zeros((2**self.N, self.N + self.M + self.N*self.M))
        for i in range(2**self.N):#iterate through spin states
            spin_state = state_from_iteration(self.N, i)
            
            db = np.array([self.wavefunction[i]*(np.tanh(np.dot(self.weights[:,j], spin_state) + self.b[j])) for j in range(self.M)])

            dw = np.array([[db[m]*spin_state[n] for m in range(self.M)] for n in range(self.N)])

            da = np.array([spin_state[k]*self.wavefunction[i] for k in range(self.N)])
            
            del_psi[i] = np.append(np.append(da,db),dw)            

        return del_psi
    def grad_e(self):
        del_psi = self.grad_psi()
        bra_wf = np.transpose(self.wavefunction)

        right = np.matmul(self.hamiltonian,del_psi)
        return 2*(np.real(np.matmul(bra_wf,right)) + self.EnergyExpectation()*np.real(np.matmul(bra_wf,del_psi)))/norm(self.wavefunction)

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
        self.updateWF()


def Train(epsilon):
    test = NeuralNet(5,5,0.1)
    training = []
    i = 0
    while(True):
        print(test.wavefunction)
        training.append(test.EnergyExpectation())
        print(training[i])
        if i>=5:
            breaker = True
            for j in range(4):
                if abs(training[len(training)-j-1] - training[len(training)-j-2]) >= epsilon:
                    breaker = False
            if breaker==True:
                return training

        test.UpdateOnce()
        i += 1
        
Train(0.0001)

