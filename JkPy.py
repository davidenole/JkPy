from matplotlib import pyplot as plt
import h5py
from math import log, sqrt
import numpy as np
from os import listdir
from os.path import isfile, join

# converts number to string of length min(4,n_figures)
def mstr(n):
    s = str(n)
    if len(s)<4:
        return '0'*(4-len(s))+s
    else:
        return s

# calculates the mean of a list of elements at position i
def listmean(L,i):
    l = [ el[i] for el in L ]
    return mean(l)

# calculates the std of a list of elements at position i
def liststd(L,i):
    l = [ el[i] for el in L ]
    return std(l)

# jackknife 1-removed sampling of a sample of trajectories
def jackknifeSampling(L):
    n = len(L)
    m = len(L[0])
    l = [ [ [el for el in L[j]] for j in range(n) if j!=i] for i in range(n)]
    return [ [ listmean(M,i) for i in range(m) ] for M in l ]

# calculates the pion mass for a jackknife sample
def pionMass(L):
    n = len(L[0])
    pm = [ [ log(l[i]/l[i+1]) for i in range(n-1) ] for l in L ]
    return pm

# calculates the pcac mass for a jackknife sample
def pcacMass(P,A):
    n = len(P[0])
    pcac = [ [ 0.25*( A[k][i+1] - A[k][i-1] )/P[k][i] for i in range(1,n-1) ] for k in range(len(P)) ]
    return pcac

# calculates the jackknife error of a sample
def error(L,n):
    return [ sqrt(n)*liststd(L,i) for i in range(len(L[0])) ]

# calculates constant fit
def constantFit(Y,S):
    w = [ 1/s/s for s in S ]
    x = [ w[i]*Y[i] for i in range(len(Y)) ]
    return sum(x)/sum(w) , (sum(w))**(-.5)
    
# read propagators from tmlqcd-like file
def readPropsTMLQCD(filename,T):
    t = T+2
    read = np.fromfile(filename, sep='  ').reshape( t//2*3, 5)[:,3:]
    PP = np.concatenate((read[:t,0],read[1:t-1,0][::-1]))
    PA = np.concatenate((read[t:2*t,0],read[t+1:2*t-1,1][::-1]))*(-1)
    PV = np.concatenate((read[2*t:3*t,0],read[2*t+1:3*t-1,1][::-1]))
    return PP, PA, PV

L = 48
T = 96

# tmlqcd
nmasses = 3
NCs = 10
N = NCs
masslist = [-0.0020,-0.0028,-0.0036]

cn = [0]*nmasses
cn[0] = [540,600,640,1283,1543,1760,1800,1803,1920,2000,2040,2063]
cn[1] = [560,1000,1080,1200,1360,1560,1680,1720,1800,2080]
cn[2] = [520,680,720,760,840,880,920,1040,1120,1160,1400,1640]

#mypath = './files/'
#files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#confs = [ f.split('.')[-1] for f in files ]

#for m in range(nmasses):
#    cn[m] = confs[ m*NCs : (m+1)*NCs+1 ]

colours = ['r','g','b']
for m in masslist:
    print( m )
    Ps = []
    PAs = []
    col = colours[masslist.index(m)]
    for ncn in cn[masslist.index(m)]:

        f = open('./files/onlinemeas.00'+mstr(ncn))
        data = f.read().split('\n')[:T+2]
        f.close()
        tPu = [0]*T
        tPAu = [0]*T
        for i in range(T//2+1):
            el = data[i]
            el2 = data[i+T//2+1]
            tPu[i] = float(el.split('  ')[-2])
            if i not in [0,T//2]:
                tPu[T-i] = float(el.split('  ')[-1])
            tPAu[i] = -float(el2.split('  ')[-2])
            if i not in [0,T//2]:
                tPAu[T-i] = -float(el2.split('  ')[-1])

        f.close()
        Ps.append(tPu)
        PAs.append(tPAu)

    # jk resampling
    Ps_jk = jackknifeSampling(Ps)
    PAs_jk = jackknifeSampling(PAs)    

    Pms = pionMass(Ps_jk)
    Pcacms = pcacMass(Ps_jk,PAs_jk)
    
    sPms = error(Pms,N-1)
    sPcacms = error(Pcacms,N-1)

    # fit PCAC

    X = [ i for i in range (20,T-20)]
    Y = [ [el[t] for t in X] for el in Pcacms ]
    W = [ 1/sPcacms[t] for t in X ]
    
    pcacMassList = [ polyfit(X, Y[i], 0, w=W)[0] for i in range(N) ]

    mpcac = mean(pcacMassList)
    smpcac = std(pcacMassList)*sqrt(N-1)

    print( mpcac, smpcac)
    
    xplot = [ t for t in range(1,T-1) ]
    yplot = [ listmean(Pcacms,i) for i in range(T-2) ]

    fitplot = [ mpcac for i in range(T-2) ]
  
    plt.errorbar( xplot, yplot, yerr=[w for w in sPcacms], fmt=col, marker="o",  markersize=2, linestyle='None', label=str(m) )
    plt.plot( xplot, fitplot, col + '--')

plt.legend()
plt.show()
