'''
Input: J, dh_

Output: h

Runs BP from matlab file, produces post-MP potential
'''
#import matlab.engine
import pymatlab
import numpy as np
#from oct2py import octave

#eng = matlab.engine.start_matlab()
def MP(J, dh_):
    #from oct2py import octave
    #y.putvalue('A, B, C', gabp(J, dh_))
    # session = pymatlab.session_factory()
    # session.run("function [Jii,hi,xmap] = gabp(J,h) eps = 1e-12; maxiter = 5; NH=zeros(size(J)); NH = abs(J)>0; for i = 1:length(h) NH(i,i)=0; end nh = logical(NH); M = zeros(size(J)); Mh = zeros(size(J)); for i = 1:length(h) M(i,:)= J(i,i); Mh(i,:)=h(i); end disp(size(h)); mo = zeros(size(M)); mh = zeros(size(Mh)); dJ=diag(J); hs = zeros(size(h)); Jii = zeros(size(h)); hi = zeros(size(h)); for iter = 1:maxiter s=zeros(size(h)); for i = 1:length(h) s(i)=sum(J(i,nh(i,:))'./(M(nh(i,:),i)).*J(nh(i,:),i)); hs(i) = sum(J(i,nh(i,:))'./(M(nh(i,:),i)).*(Mh(nh(i,:),i))); Jii(i)=dJ(i)-s(i); mo(i,nh(i,:))=Jii(i)+(J(i,nh(i,:))'./M(nh(i,:),i).*J(nh(i,:),i))'; hi(i) = h(i)-hs(i);  mh(i,nh(i,:))=hi(i)+(J(i,nh(i,:))'./M(nh(i,:),i).*Mh(nh(i,:),i))'; end xmap = hi./Jii; if (sum(abs(M(:)-mo(:)))<eps) return end M=mo;Mh = mh; end disp(xmap); disp(M);")
    # session.putvalue('J', J)
    # session.putvalue('dh_', dh_)
    # session.run('A, B, C = gabp(J, dh_)')
    # result = session.getValue('A')
    #ret = eng.gabp(J, dh_)
    A, B, C = gabp(J, dh_)
    print(A, B, C)
    return B

def gabp(J, h):
    eps = 1e-12
    maxiter = 10000
    NH = np.zeros(J.shape)
    NH = abs(J) > 0
    for i in range(h.shape[0]):
        NH[i, i] = 0
    nh = NH != 0
    M = np.zeros(J.shape)
    Mh = np.zeros(J.shape)
    for i in range(h.shape[0]):
        M[i, :] = J[i, i]
        Mh[i, :] = h[i]
    mo = np.zeros(J.shape)
    mh = np.zeros(J.shape)
    dJ = np.diag(J)
    hs = np.zeros(h.shape)
    Jii = np.zeros(h.shape)
    hi = np.zeros(h.shape)
    for iter in range(maxiter):
        s = np.zeros(h.shape)
        for i in range(h.shape[0]):
            s[i] = np.sum(J[i,nh[i,:]].T/M[nh[i,:], i]*J[nh[i,:], i])
            hs[i] = np.sum(J[i, nh[i, :]].T/ M[nh[i, :], i] * Mh[nh[i, :], i])
            Jii[i] = dJ[i] - s[i]
            mo[i, nh[i,:]]= Jii[i] + J[i, nh[i,:]].T /M[nh[i,:],i]*J[nh[i,:],i]
            hi[i] = h[i] - hs[i]
            mh[i, nh[i, :]] = hi[i] + J[i, nh[i,:]].T /M[nh[i,:],i]*Mh[nh[i,:],i]
        xmap = hi / Jii
        if np.sum(abs(M[:]-mo[:])) < eps:
            return Jii,hi,xmap
        M,Mh = mo, mh

    return Jii, hi, xmap
