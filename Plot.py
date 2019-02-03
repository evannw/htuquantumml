import numpy as np
import matplotlib.pyplot as plt

def plot(N, spinCorrelations, magnetizations, energyExpectations):
    SC = open(spinCorrelations, "r")
    MZ = open(magnetizations, "r")
    EE = open(energyExpectations, "r")
    energies_list = np.load(EE)
    mag_list = np.load(MZ)
    corr_list = np.load(SC)
    plt.figure(0)
    for EE in energies_list:
        finalEnergy = np.around(EE[len(EE)-1],decimals=2)
        plt.annotate(str(finalEnergy), (len(EE), finalEnergy))
        plt.plot(EE)
    plt.title('Energy During Training')
    plt.xlabel('Iteration')
    #plot magnetizations
    fig, axarr = plt.subplots(nrows=3,sharex=True)
    for i in range(3):
        axarr[i].bar(range(N),mag_list[i])
        axarr[i].set_ylabel(['X','Y','Z'][i] + ' Magnetization')
        plt.locator_params(nbins = len(mag_list[i]))
    plt.xlabel('Site')
    fig.suptitle('Magnetization of Sites')

    #plot correlations
    fig, axarr = plt.subplots(nrows=3,sharex=True)
    xs = [i for i in range(len(corr_list[0]) + 1) if i != test_site]
    for i in range(3):
        axarr[i].bar(xs,corr_list[i])
        axarr[i].set_ylabel(['X','Y','Z'][i] + ' Correlation')
        plt.locator_params(nbins = len(corr_list[i]))
    plt.xlabel('Site')
    fig.suptitle('Correlation of Site ' + str(test_site) + ' with Other Sites')

    plt.show()

plot(6,"SpinCorrelations.npy","Magnetizations.npy","EnergyExpectations.npy")