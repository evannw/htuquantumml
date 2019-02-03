import numpy as np

# import matplotlib.pyplot as plt

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

#Returns a Hamiltonian diagonal matrix of size 2^nx2^n using T_Ising model
def T_ising(N,J,h):
    valStore = np.zeros((2**N,2**N))
    finalVal = np.zeros((2**N,2**N))
    for i in range(N):
        singleCurrent = np.array([1])
        for j in range(N):
            current = np.array([1])
            for k in range(N):
                if k!=i and k!=j:
                    current = np.kron(current, np.identity(2))
                    # valList.append(numpy.identity(2))
                elif k == i:
                    current = np.kron(current, sz)
                else:
                    current = np.kron(current, sz)
            if j!=i:
                singleCurrent = np.kron(singleCurrent, np.identity(2))
            else:
                singleCurrent = np.kron(singleCurrent, sx)
            
            valStore += J*current
        finalVal += h*singleCurrent
    return valStore+finalVal

def Hamiltonian_heisenberg(N):
    ham = np.zeros((2**N,2**N))
    for i in range(N-1):
        product_terms = [np.identity(2) for j in range(N)]
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

def correlation(site,wavefunction,direction):
    N = int(np.log2(len(wavefunction)))
    product_terms = [np.identity(2) for i in range(N)]
    if direction == 'x':
        product_terms[site] = sx
    elif direction == 'y':
        product_terms[site] = sy
    elif direction == 'z':
        product_terms[site] = sz

    product_list = []
    for i in range(N):
        if i == site:
            continue
        product_terms[i] = product_terms[site]
        
        prod = np.ones(1)
        for term in product_terms:
            prod = np.kron(prod,term)
        prod = np.dot(np.conjugate(wavefunction),np.matmul(prod,wavefunction))
        product_list.append(prod)
        product_terms[i] = np.identity(2)
    return product_list

def T_isingWrapper(N):
    return T_ising(N, 1, 0.001)

def magnetization(N):
    valStoreX = []
    valStoreY = []
    valStoreZ = []
    for i in range(N):
        singleCurrentX = np.array([1])
        singleCurrentY = np.array([1])
        singleCurrentZ = np.array([1])
        for j in range(N):
            if j!=i:
                singleCurrentX = np.kron(singleCurrentX, np.identity(2))
                singleCurrentY = np.kron(singleCurrentY, np.identity(2))
                singleCurrentZ = np.kron(singleCurrentZ, np.identity(2))
            else:
                singleCurrentX = np.kron(singleCurrentX, sx)
                singleCurrentY = np.kron(singleCurrentY, sy)
                singleCurrentZ = np.kron(singleCurrentZ, sz)
        valStoreX.append(singleCurrentX)
        valStoreY.append(singleCurrentY)
        valStoreZ.append(singleCurrentZ)
    return valStoreX,valStoreY,valStoreZ



class NeuralNet(object):

    def __init__(self, N, M, D, h_function):
        self.N = N
        self.M = M
        self.a = np.random.random(N)
        self.b = np.random.random(M)
        #D is learning rate (used in update)
        self.D = D
        #Computes a generic hamiltonian matrix for size N
        self.hamiltonian = h_function(self.N)
        self.wavefunction = []
        self.weights = np.random.rand(N,M) + 1j*np.random.rand(N,M)
        # self.weights = np.random.normal(N,M)
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
        bra_wf = np.conjugate(self.wavefunction)

        right = np.matmul(self.hamiltonian,del_psi)
        return 2*(np.real(np.matmul(bra_wf,right)) - self.EnergyExpectation()*np.real(np.matmul(bra_wf,del_psi)))/norm(self.wavefunction)

    def EnergyExpectation(self):
        return np.dot(np.conjugate(self.wavefunction),np.matmul(self.hamiltonian,self.wavefunction))/norm(self.wavefunction)
    
    def MagnetizationEE(self):
        operatorX, operatorY, operatorZ = magnetization(self.N)
        operatorX = np.real([np.dot(np.conjugate(self.wavefunction),np.matmul(x,self.wavefunction))/norm(self.wavefunction) for x in operatorX])
        operatorY = np.real([np.dot(np.conjugate(self.wavefunction),np.matmul(y,self.wavefunction))/norm(self.wavefunction) for y in operatorY])
        operatorZ = np.real([np.dot(np.conjugate(self.wavefunction),np.matmul(z,self.wavefunction))/norm(self.wavefunction) for z in operatorZ])
        return operatorX, operatorY, operatorZ

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

#Train until energy converges to within epsilon
def TrainEpsilon(N, M, rate, epsilon, hamiltonian):
    test = NeuralNet(N,M, rate, hamiltonian)
    trainingEE = []
    i = 0
    while(True):
        # print(test.wavefunction)

        trainingEE.append(test.EnergyExpectation())
        if i>=5:
            breaker = True
            for j in range(4):
                if abs(trainingEE[len(trainingEE)-j-1] - trainingEE[len(trainingEE)-j-2]) >= epsilon:
                    breaker = False
            if breaker==True:
                MEEx, MEEy, MEEz = test.MagnetizationEE()
                return trainingEE, test.wavefunction, MEEx, MEEy, MEEz
        test.UpdateOnce()
        i += 1

#Train over a certain number of iterations
def TrainIterations(N, M, rate, iterations, hamiltonian):
    test = NeuralNet(N,M, rate, hamiltonian)
    trainingEE = []
    for i in range(iterations):
        # print(test.wavefunction)
        trainingEE.append(test.EnergyExpectation())
        test.UpdateOnce()
    MEEx, MEEy, MEEz = test.MagnetizationEE()
    return trainingEE, test.wavefunction, MEEx, MEEy, MEEz

def measure(N,M,rate,iterations,test_site,h_function):
    """SC = open("SpinCorrelations.txt", "w")
    MZ = open("Magnetizations.txt", "w")
    EE = open("EnergyExpectations.txt", "w")"""
    corr_list = np.zeros((iterations,3,N-1)) #3 represents the x, y, and z correlations
    mag_list = np.zeros((iterations,3,N))
    #plt.figure(0)
    corr_list = np.zeros((iterations,3,N-1))#3 represents the x, y, and z correlations
    mag_list = np.zeros((iterations,3,N))
    energies_list = [[] for i in range(iterations)]
    max_len = 0
    for n in range(iterations):
        energies, wf, MEEx, MEEy, MEEz = np.real(TrainIterations(N,M,rate,2000,h_function))
        corr_list[n] = np.array([correlation(test_site,wf,s) for s in ['x','y','z']])
        mag_list[n] = np.array([MEEx,MEEy,MEEz])
        #plt.plot(EE)
        # finalEnergy = np.around(EE[len(EE)-1],decimals=2)
        # plt.annotate(str(finalEnergy), (len(EE), finalEnergy))
        energies_list[n] = energies
        if len(energies) > max_len:
            max_len = len(energies)
    #EE.write("[")
    for i in range(len(energies_list)):
        last = energies_list[i][-1]
        energies_list[i] = energies_list[i] + [last for j in range(max_len - len(energies_list[i]))]
        """if i != 0:
            EE.write("), array(" + str(energies_list[i]))
        else:
            EE.write("array(" + str(energies_list[0]))
    EE.write(")]")"""
    mag_list = np.sum(mag_list,axis = 0)/iterations#average magnetizations
    #MZ.write(str(mag_list))
    corr_list = np.sum(corr_list,axis=0)/iterations#calculate average correlations
    #SC.write(str(corr_list))
    np.save('mag_list.npy',mag_list)
    np.save('corr_list.npy',corr_list)
    np.save('energies_list',np.array(energies_list))
    #EE.close()
    #MZ.close()
    #SC.close()
    # plt.figure(0)
    # for EE in energies_list:
    #     plt.plot(EE)
    # plt.title('Energy During Training')
    # plt.xlabel('Iteration')
    # #plot magnetizations
    # fig, axarr = plt.subplots(nrows=3,sharex=True)
    # for i in range(3):
    #     axarr[i].bar(range(N),mag_list[i])
    #     axarr[i].set_ylabel(['X','Y','Z'][i] + ' Magnetization')
    #     plt.locator_params(nbins = len(mag_list[i]))
    # plt.xlabel('Site')
    # fig.suptitle('Magnetization of Sites')

    # #plot correlations
    # fig, axarr = plt.subplots(nrows=3,sharex=True)
    # xs = [i for i in range(len(corr_list[0]) + 1) if i != test_site]
    # for i in range(3):
    #     axarr[i].bar(xs,corr_list[i])
    #     axarr[i].set_ylabel(['X','Y','Z'][i] + ' Correlation')
    #     plt.locator_params(nbins = len(corr_list[i]))
    # plt.xlabel('Site')
    # fig.suptitle('Correlation of Site ' + str(test_site) + ' with Other Sites')

    # plt.show()

measure(14,7,0.01,5,0,Hamiltonian_heisenberg)
