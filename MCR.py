# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: admin
"""

import numpy as np
from matplotlib.pyplot import show, plot, text
from NetCDF import netcdf_reader,plot_ms,plot_tic
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.metrics import explained_variance_score
from scipy.linalg import norm

def back_remove(xx,point,range_point):
    xn=list(np.sum(xx,1))
    n1=xn.index(min(xn[0:range_point-point]))
    n3=xn.index(min(xn[xx.shape[0]-range_point+point:xx.shape[0]]))

    if n1<range_point-point/2:
        n2 = n1+3
    else:
        n2 = n1-3
    if n3<xx.shape[0]-range_point-point/2:
        n4 = n3+3
    else:
        n4 = n3-3
    Ns = [[min(n1,n2),max(n1,n2)],[min(n3,n4),max(n3,n4)]]

    bak = np.zeros(xx.shape)
    for i in range(0, xx.shape[1]):
        tiab = []
        reg = []
        for j in range(0, len(Ns)):
            tt = range(Ns[j][0],Ns[j][1])
            tiab.extend(xx[tt, i])
            reg.extend(np.arange(Ns[j][0], Ns[j][1]))
        rm = reg - np.mean(reg)
        tm = tiab - np.mean(tiab)
        b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s = np.mean(tiab)-np.dot(np.mean(reg), b)
        b_est = s+b*np.arange(xx.shape[0])
        bak[:, i] = xx[:, i]-b_est   
    bias = xx-bak
    return bak, bias


def FR(x, s, o, z, com):
    xs = x[s,:]   
    xs[xs<0]=0
    xz = x[z,:]
    xo = x[o,:]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))

    u, s0, v = np.linalg.svd(xc)
    t = np.dot(u[:,0:com],np.diag(s0[0:com]))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    u1, s1, v1 = np.linalg.svd(x)
    t1 = np.dot(u1[:, 0:com], np.diag(s1[0:com]))
    c = np.dot(t1, r)

    c1, ind = contrain_FR(c, s, o)
    c1[c1<0]=0
    spec = x[s[ind],:]

    if c1[s[ind]] == 0:
        pu = 1e-6
    else:
        pu = c1[s[ind]]
        
    cc = c1/pu

    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    left_x = x - res_x
    return cc, res_x

def contrain_FR(c, s, o):
    ind_s = np.argmax(np.abs(c[s]))

    if c[s][ind_s] < 0:
        c = -c
    
    if s[0]<o[0]:
        if c[s[-2]]<c[s[-1]]:
            ind1 = s[-1]
            ind2 = o[np.argmax(c[o])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = o[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = o[np.argmax(c[o])]
            ind2 = s[0]
        else:
            ind1 = o[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1:len(c)] = 0
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s

def mcr_by_fr(x,p,opt,peak,fragment):

    p1 = p[opt,0];p2 = p[opt,1];p3 = p[opt,2];p4 = p[opt,3]
    
    if p1==p2:
        p2=p2+1
    s = list(range(0,min(int(p1),int(p2))))
    o = list(range(min(int(p1),int(p2)),max(int(p1),int(p2))))
    z = list(range(max(int(p1),int(p2)),x.shape[0])) 
    
    cc1, xx1 = FR(x, s, o, z, 3)  

    if p3==p4:
        p4=p4+1
    s = list(range(max(int(p3),int(p4)),x.shape[0]))
    o = list(range(min(int(p3),int(p4)),max(int(p3),int(p4))))
    z = list(range(0,min(int(p3),int(p4))))                 
    
    cc3, xx3 = FR(x, s, o, z, 3)

    xx2 = x-xx1-xx3
    
    ind_s = np.argmax(np.abs(np.sum(xx2,1)))

    cc2 = ittfa(xx2, ind_s, 1)

    S = np.zeros((1, x.shape[1]))
    for j in range(0, S.shape[1]):
        a = fnnls(np.dot(cc2.T, cc2), np.dot(cc2.T, xx2[:, j]), tole='None')
        S[:, j] = a['xx']

    xx2 = np.dot(cc2,S)

    re_x = xx1+xx2+xx3
    
    re_chrom  = np.zeros((len(peak), x.shape[0]))
    re_chrom[0,:] = np.sum(xx1,1)
    re_chrom[1,:] = np.sum(xx2,1)
    re_chrom[2,:] = np.sum(xx3,1)
    
    R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')

    return re_chrom,R2
    
def get_fragment(y):
    fragment = np.zeros((y.shape[0],2))

    for i in range(y.shape[0]):
        for k in range(int(y.shape[1]-5)):
            if np.mean(y[i,k:k+5])>0.9:
                fragment[i,0] = k
                n = 4
                while y[i,k+n]>=0.9:
                    n += 1
                    if y[i,k+n]<0.9:
                        fragment[i,1] = k+n
                        break
                break
    return fragment
    
def Peak_detection(fragment,point): 
    peak = []
    for i in range(fragment.shape[0]):
        if fragment[i,0]==0:
            continue
        else:
            peak_i = [i]
            for j in range(fragment.shape[0]):
                if j!=i and fragment[j,0]!=0:

                    if ((fragment[i,1]+2*point < fragment[j,0]) or
                        (fragment[i,0]-2*point > fragment[j,1])):
                        continue
                    else:
                        peak_i.append(j)
                else:
                    continue
            peak.append(peak_i)    

    peak1 = peak
    for h in range(fragment.shape[0]): 
        ind = []
        for k in range(len(peak)): 
            if h in peak[k]:
                ind.append(k)
        composition=[]
        for q in range(len(ind)):            
            composition = set(composition).union(set(peak[ind[q]]))
        for p in range(len(ind)): 
            peak1[ind[p]] = list(composition)
    
    peak = []
    for k in range(len(peak1)):
        peak1[k].sort()
        if peak1[k] not in peak:
            peak.append(peak1[k])
    return peak

def optim_frag(peak0,peak1,xshape,point,range_point):
       
    pp1 = abs(peak0[1] - peak0[0]) + range_point
    pp2 = abs(peak1[0] - peak0[0]) + range_point 
    pp3 = abs(peak0[2] - peak0[0]) + range_point 
    pp4 = abs(peak1[1] - peak0[0]) + range_point
    pp = [pp1,pp2,pp3,pp4]
    pp = np.tile(pp,(point**4,1))
    p1 = [];p2 = [];p3 = [];p4 = []  
    for j in range(1-point,1):
        for p in range(0,point):
            for q in range(1-point,1):
                for m in range(0,point):    
                    p1.append(j);p2.append(p);p3.append(q);p4.append(m)
    p = np.zeros((point**4,4))    
    p[:,0] = p1;p[:,1] = p2;p[:,2] = p3;p[:,3] = p4
    p = p+pp
                
    return p

def ittfa(d, needle, pcs):
    u, s, v = np.linalg.svd(d)
    t = np.dot(u[:,0:pcs], np.diag(s[0:pcs]))
    row = d.shape[0]
    cin = np.zeros((row, 1))
    cin[needle-1] = 1
    out = cin
    for i in range(0, 100):
        vec = out
        out = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), out)
        out[out < 0] = 0
        out = unimod(out, 1.1, 2)
        out = out/norm(out)
        kes = norm(out-vec)
        if kes < 1e-6 or iter == 99:
            break
    return out

def unimod(c, rmod, cmod, imax=None):
    ns = c.shape[1]
    if imax == None:
        imax = np.argmax(c, axis=0)
    for j in range(0, ns):
        rmax = c[imax[j], j]
        k = imax[j]
        while k > 0:
            k = k-1
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 0 
                    if cmod == 1:
                        c[k, j] = c[k+1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k+1, j])/2
                            c[k+1, j] = c[k, j]
                            k = k+2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
        rmax = c[imax[j], j]
        k = imax[j]

        while k < c.shape[0]-1:
            k = k+1
            if k==53:
                k=53
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k-1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k-1, j])/2
                            c[k-1, j] = c[k, j]
                            k = k-2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
    return c

def fnnls(x, y, tole):
    xtx = np.dot(x, x.T)
    xty = np.dot(x, y.T)
    if tole == 'None':
        tol = 10*np.spacing(1)*norm(xtx, 1)*max(xtx.shape)
    mn = xtx.shape
    P = np.zeros(mn[1])
    Z = np.array(range(1, mn[1]+1), dtype='int64')
    xx = np.zeros(mn[1])
    ZZ = Z-1
    w = xty-np.dot(xtx, xx)
    iter = 0
    itmax = 30*mn[1]
    z = np.zeros(mn[1])
    while np.any(Z) and np.any(w[ZZ] > tol):
        t = ZZ[np.argmax(w[ZZ])]
        P[t] = t+1
        Z[t] = 0
        PP = np.nonzero(P)[0]
        ZZ = np.nonzero(Z)[0]
        nzz = np.shape(ZZ)
        if len(PP) == 1:
            z[PP] = xty[PP]/xtx[PP, PP]
        elif len(PP) > 1:
            if np.linalg.det(xtx[np.ix_(PP, PP)]) ==0:
                small = 1e-6*np.identity(xtx[np.ix_(PP, PP)].shape[0])
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]+small))
            else:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
        z[ZZ] = np.zeros(nzz)
        while np.any(z[PP] <= tol) and iter < itmax:
            iter += 1
            qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
            alpha = np.min(xx[qq] / (xx[qq] - z[qq]))
            xx = xx + alpha*(z - xx)
            ij = np.nonzero(tuple(np.abs(xx) < tol) and tuple(P != 0))
            Z[ij[0]] = ij[0]+1
            P[ij[0]] = np.zeros(max(np.shape(ij[0])))
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = np.shape(ZZ)
            if len(PP) == 1:
                z[PP] = xty[PP]/xtx[PP, PP]
            elif len(PP) > 1:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
        xx = np.copy(z)
        w = xty - np.dot(xtx, xx)
    return{'xx': xx, 'w': w}