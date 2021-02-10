import numpy as np 
import h5py 
from mendeleev import element

def GetAbundancesFixedYef(Ytot, T9, rho, lnYef, xi, useNeutral = False):
    """
    For numpy arrays of total elemental abundances Ytot, temperatures T9 (in 
    GK), densities rho (in g/cc), and natural log of the free electron fractions
    lnYef for a set of times, calculate the abundances of the ionization states 
    defined by the ionization potentials given in the array xi 
    
    useNeutral determines which abundance to scale relative to, probably always 
    want to keep this set to False except for testing purposes.
    
    returns an array YI(len(Ytot), len(xi)+1) of the abundances 
    """
    
    # Define some constants
    Na = 6.02e23 # Avogadros number
    me = 9.109e-28 # Electron mass in grams 
    Kb = 1.3806e-16 # Boltzmann's constants in erg/K
    Kbev = Kb/1.602e-12 # Boltzmann's constant in ev/K
    hbar = 1.0545e-27 # Planck's constant in erg s

    # Calculate the log of the g function for all ionization states except for 
    # the fully ionized state 
    #lng = np.zeros((len(T9), len(xi))) 
    #for (i, x) in enumerate(xi):
    #    lng[:,i] = np.log(2/(rho*Na)*(me*Kb*1.e9*T9/(2*np.pi*hbar*hbar))**(3/2)) 
    #    lng[:,i] += - x/(Kbev*1.e9*T9) - lnYef
    lng = np.outer(np.log(2/(rho*Na)*(me*Kb*1.e9*T9/(2*np.pi*hbar*hbar))**(3/2))
        -lnYef, np.ones(len(xi)))
    lng = lng - np.outer(1.0/(Kbev*1.e9*T9), xi) 

    # Calculate the sum of ln g for every ionization state 
    lnh = np.zeros((len(T9), len(xi)+1)) 
    lnh[:,0] = 0.0
    for (i, x) in enumerate(xi):
        lnh[:,i+1] = lnh[:,i] + lng[:,i]

    # Find the maximum value and ionization state index of h for each time 
    idxMax = np.argmax(lnh, 1)
    lnhMax = np.amax(lnh, 1)

    # Work in terms of the unionized state 
    if useNeutral:
        idxMax = np.zeros(T9.shape) 
        lnhMax = np.zeros(T9.shape)
    
    # Find the scaled h factors for converting from maximum ionization state 
    #abundance to any other abundance 
    lnhScaled = lnh 
    for i in range(len(xi)+1):
        lnhScaled[:,i] = lnh[:,i] - lnhMax[:]

    # Calculate the abundance of the maximum ionization state 
    Ymax = Ytot/np.sum(np.exp(lnhScaled), 1)  

    # Calculate the abundance of all other ionization states based on maximum 
    #ionization state
    #YI = Ymax
    YI = np.zeros((len(T9), len(xi)+1)) 
    for i in range(len(xi)+1): 
        YI[:,i] = Ymax[:]*np.exp(lnhScaled[:,i])
    return YI 

def GetYef(Ytot, T9, rho, xi, niter=25, lnYeMin=-400.0): 
    """ 
    Calculates the abundances of various ionization states by self-consistently 
    finding the free electron fraction assuming charge neutrality 
    
    We use bisection since it is simple to use on the entire numpy array at the 
    same time.
    """
    
    # Set the initial range of allowed electron fractions
    lnYefLow = lnYeMin*np.ones(T9.shape) 
    lnYefHi = np.zeros(T9.shape) 
    def GetYefContribution(Ytot, T9, rho, lnYeg, xi):
        """
        For a given guess for the free electron fraction, calculate the free 
        electron fraction contributed by an element with total abundance Ytot 
        and ionization energies xi assuming charge neutrality
        """
        YefContribution = 0.0
        for (i, x) in enumerate(xi):
            YI = GetAbundancesFixedYef(Ytot[:,i], T9, rho, lnYeg, x)
            YefContribution += np.dot(YI, np.arange(0, len(x)+1, 1))
        return YefContribution
    
    # Build function to compare Yef determined by charge neutrality and by the imposed Yef 
    # Find real solution by finding roots of this equation
    fLow = np.log(GetYefContribution(Ytot, T9, rho, lnYefLow, xi)) - lnYefLow 
    fHi = np.log(GetYefContribution(Ytot, T9, rho, lnYefHi, xi)) - lnYefHi
     
    # We need to flag places where our function doesnt have opposite signs on either end of the interval 
    bad = np.where(fLow*fHi>0, 1.0, 0.0)
    
    # Perform niter bisection iterations
    for i in range(niter):
        # Take the midpoint of the lnYef range and calculate the charge neutrality electron fraction 
        # and calculate value of function we want to find the root of 
        lnYefMid = 0.5*(lnYefLow + lnYefHi) 
        fMid = np.log(GetYefContribution(Ytot, T9, rho, lnYefMid, xi)) - lnYefMid
        
        # Replace either the low or high end of the range with the midpoint value depending on 
        # the signs of the root function 
        lnYefLow = np.where(fMid*fLow>0, lnYefMid, lnYefLow)
        fLow = np.where(fMid*fLow>0, fMid, fLow)
        
        lnYefHi = np.where(fMid*fHi>0, lnYefMid, lnYefHi)
        fHi = np.where(fMid*fHi>0, fMid, fHi)
        
    lnYefMid = 0.5*(lnYefLow + lnYefHi)
    
    # Choose the minimum value of Ye when the root doesn't sit in the initial range
    # This is safe since the upper bound is truly an upper bound 
    lnYefMid = np.where(bad>0.5, lnYeMin, lnYefMid)
    
    # Return the free electron fraction
    return (np.exp(lnYefMid), GetYefContribution(Ytot, T9, rho, lnYefMid, xi))

class SkyNetIonization: 
    def __init__(self, fname, calculateYeFree = True): 
        f = h5py.File(fname,'r')
        
        Z = np.array(f['Z'])
        self.time = np.array(f['Time'])
        self.rho = np.array(f['Density'])
        
        # Read in the temperature and replace with a power law expansion
        T9Orig = np.array(f['Temperature'])
        idx = np.searchsorted(-T9Orig,-1.e-3)
        self.T9 = T9Orig
        self.T9[idx:] = self.T9[idx]*(self.time[idx]/self.time[idx:])
        
        # Calculate elemental abundances Ytot 
        nElements = np.max(Z)+1
        nTimes = self.T9.shape[0]
        Y = np.array(f['Y'])
        self.Ytot = np.zeros((nTimes, nElements))
        for (i, zz) in enumerate(Z):
            self.Ytot[:, zz] += Y[:, i]  
        
        # Calculate total Ye from SkyNet abundances 
        self.YeTot = np.zeros(self.T9.shape) 
        for zz in range(nElements):
            self.YeTot += zz*self.Ytot[:,zz]
        
        # Tabulate ionization potentials using the Mendeleev library 
        self.xi = [] 
        self.xi.append([]) # Account for neutrons
        for zz in range(1,nElements):
            self.xi.append(np.array(list(element(zz).ionenergies.values()))) # Use ionization energies from Mendeleev package
        
        
        if (calculateYeFree):
            self.ConsistentYeFreeSet = True 
            self.Yef = self.CalculateSelfConsistentYef()
        else: 
            self.ConsistentYeFreeSet = False
            self.Yef = self.YeTot
            
    def CalculateSelfConsistentYef(self):
        (Yef, YefSum) = GetYef(self.Ytot, self.T9, self.rho, self.xi) 
        return Yef

    def GetIonizationStateAbundancesElement(self, zz):
        return GetAbundancesFixedYef(self.Ytot[:,zz], self.T9, self.rho, 
                                     np.log(self.Yef), self.xi[zz])
   
    def GetIsoElectronicAbundances(self, zmin, zmax): 
        YI = np.zeros((self.T9.shape[0],zmax+1))
        for zz in range(zmin, zmax+1):
            YI[:,zmax-zz:] += self.GetIonizationStateAbundancesElement(zz)
        return YI