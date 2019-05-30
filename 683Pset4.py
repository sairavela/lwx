import numpy as np
import matplotlib.pyplot as plt
num = 100000

#Problem 3.a

v = np.random.uniform(low=0, high=1, size=(num,1))
def bid(v_t, n):
    return (1-1.0/n)*v_t #From Lecture for U(0, 1) distribution

#Problem 3.b

N = 4
groups = np.reshape(v, (int(num/N), N))
bidsFunction = np.vectorize(lambda x: bid(x, N), otypes=[np.float])
bids = bidsFunction(groups)

#Problem 3.c

def kernel(u, type):
    if type=='uniform':
        if abs(u) <= 1:
            return .5
        else:
            return 0
    elif type=='epanechnikov':
        if abs(u) <= 1:
            return .75*(1-u**2)
        else:
            return 0
#Estimate G:
def G(X, z):
    def help(x_i, z):
        if x_i <= z:
            return 1
        else:
            return 0
    helpFunction = np.vectorize(lambda x: help(x, z), otypes=[np.float])
    Z = helpFunction(X)
    return (1/X.size)*Z.sum()
X = [x/100.0 for x in range(0, 76)]
trueCDF = [x/75.0 for x in range(0, 76)]
yCDF = [G(bids, x) for x in X]
#plot G
plt.title('CDF Estimation')
plt.plot(X, yCDF)
plt.plot(X, trueCDF, ':')
plt.savefig('unrelated/cdfEstimation'+str(num)+'.png')
#Estimate g:
def g(X, z, h, type):
    helpFunction = np.vectorize(lambda x: kernel((x-z)/h, type), otypes=[np.float])
    Z = helpFunction(X)
    return (1/(X.size*h))*Z.sum()
plt.close()
#Plot g
H = [.5, .1, .05, .01]
plt.title('PDF Estimation: Uniform Kernel')
X = [x/100.0 for x in range(0, 76)]
truePDF = [1/.75 for x in range(0, 76)]
for h in H:
    yCDF = [g(bids, x, h, 'uniform') for x in X]
    plt.plot(X, yCDF, label='h = '+str(h))
plt.plot(X, truePDF, ':') #True Line
plt.legend()
plt.savefig('unrelated/pdfUniform'+str(num)+'.png')
plt.close()
plt.title('PDF Estimation: Epanechnikov Kernel')
X = [x/100.0 for x in range(0, 76)]
truePDF = [1/.75 for x in range(0, 76)]
for h in H:
    yCDF = [g(bids, x, h, 'epanechnikov') for x in X]
    plt.plot(X, yCDF, label='h = '+str(h))
plt.plot(X, truePDF, ':')
plt.legend()
plt.savefig('unrelated/pdfEpanechnikov'+str(num)+'.png')
plt.close()
#Problem 3d
def inversion(bids, b_t, h, type):
    return b_t + G(bids, b_t)/((N-1)*g(bids, b_t, h, type))

b = bidsFunction(v)
#Uniform graph
'''plt.title('Value Estimation: Uniform Kernel')
X = [x/100.0 for x in range(0, 101)]
#truePDF = [1/.75 for x in range(0, 76)]
for h in H:
    invertFunction = np.vectorize(lambda x: inversion(b, x, h, 'uniform'), otypes=[np.float])
    plt.scatter(v.tolist(), invertFunction(b).tolist(), label='h = '+str(h))
    print(str(h)+': '+str(np.sum((v-invertFunction(b))**2)))
plt.plot(X, X, ':') #True Line
plt.legend()
plt.savefig('unrelated/valueUniform'+str(num)+'.png')
plt.close()
plt.title('Value Estimation: Epanechnikov Kernel')
X = [x/100.0 for x in range(0, 101)]
#truePDF = [1/.75 for x in range(0, 76)]
for h in H:
    invertFunction = np.vectorize(lambda x: inversion(b, x, h, 'epanechnikov'), otypes=[np.float])
    plt.scatter(v.tolist(), invertFunction(b).tolist(), label='h = '+str(h))
    print(str(h)+': '+str(np.sum((v-invertFunction(b))**2)))
plt.plot(X, X, ':') #True Line
plt.legend()
plt.savefig('unrelated/valueEpanechnikov'+str(num)+'.png')
plt.close()'''
#Problem 3e
vHat = np.vectorize(lambda x: inversion(b, x, .01, 'epanechnikov'), otypes=[np.float])(b)
#Estimate F:
X = [x/100.0 for x in range(0, 101)]
trueCDF = [x/100.0 for x in range(0, 101)]
yCDF = [G(vHat, x) for x in X]
#plot G
plt.title('CDF Estimation')
plt.plot(X, yCDF)
plt.plot(X, trueCDF, ':') #True Line
plt.savefig('unrelated/FcdfEstimation'+str(num)+'.png')
#Estimate g:
plt.close()
#Plot g
H = [.5, .1, .05, .01]
plt.title('PDF Estimation: Uniform Kernel')
X = [x/100.0 for x in range(0, 101)]
truePDF = [1/1 for x in range(0, 101)]
for h in H:
    yCDF = [g(vHat, x, h, 'uniform') for x in X]
    plt.plot(X, yCDF, label='h = '+str(h))
plt.plot(X, truePDF, ':') #True Line
plt.legend()
plt.savefig('unrelated/FpdfUniform'+str(num)+'.png')
plt.close()
plt.title('PDF Estimation: Epanechnikov Kernel')
X = [x/100.0 for x in range(0, 101)]
truePDF = [1/1 for x in range(0, 101)]
for h in H:
    yCDF = [g(vHat, x, h, 'epanechnikov') for x in X]
    plt.plot(X, yCDF, label='h = '+str(h))
plt.plot(X, truePDF, ':') #True Line
plt.legend()
plt.savefig('unrelated/FpdfEpanechnikov'+str(num)+'.png')
plt.close()