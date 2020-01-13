# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: admin
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import os
import csv
from NetCDF import netcdf_reader
from MCR import back_remove,ittfa,fnnls,get_fragment,Peak_detection,optim_frag,mcr_by_fr
from sklearn.metrics import explained_variance_score
import datetime
from scipy.stats import pearsonr
from scipy.linalg import norm

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
    
def preprocess(X_raw,channle): 
    X = np.zeros(X_raw.shape)
    for r in range(X.shape[0]):
        X[r,:] = 10000*X_raw[r,:]/np.max(X_raw[r,:])    

    size = int(channle/2)
    Xs =  np.zeros((size,X.shape[1]))
         
    q = int(channle*X.shape[1])
    X_p = np.zeros((X.shape[0],q))
    Xnew = np.vstack((Xs,X,Xs))
    for k in range(X.shape[0]):
        Xk = Xnew[k:k+channle,:].reshape(1,q)
        X_p[k,:] = Xk  
              
    return X_p

def Hot_map(RT,y,path):  
    xLabel = RT
    yLabel = list(range(1,y.shape[0]+1))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel('Retention time')
    ax.set_ylabel('Compound')
    im = ax.imshow(y,interpolation='none', cmap=plt.cm.hot_r)
    position=fig.add_axes([0.92, 0.37, 0.02, 0.25])
    plt.colorbar(im,cax=position)
    plt.show()
    plt.savefig(path+'/HOT.png')
    
def plot_chrom(x,re_chrom,peak,method,path):

    plt.figure(figsize = (8,6))

    plt.subplot(211)
    plt.ylim(0, 1.1*np.max(np.sum(x,1)))
    plt.plot(np.sum(x,1),linewidth=2,label='Raw_chrom')
    plt.plot(np.sum(re_chrom,0),linewidth=2,label='ReCon_chrom')  
    
    plt.ylabel('Intensity',size=15)
    
    loc=1;ncol=1     
    plt.legend(loc=loc,ncol=ncol,prop={'size':8})
    
    plt.subplot(212)
    plt.ylim(0, 1.1*np.max(re_chrom))
    for i in range(len(peak)):  
        plt.plot(re_chrom[i,:],linewidth=2,label='Componend_'+str(int(peak[i]+1)))
    loc=1;ncol=1 
    plt.xlabel('Scans',size=15)
    plt.ylabel('Intensity',size=15)
    plt.legend(loc=loc,ncol=ncol,prop={'size':8})
    plt.savefig(path+'/peak_'+str(peak)+'_by_'+str(method)+'.png')
      
def MCR(components,X,RT,y,result_path): 
    
    component = []
    retention_time  = []
    overlapping_num = []
    R_R2 = []
    r_r2 = [] 
    MCR_method = [] 
    point = 4
    range_point = 10
    print('Get the fragments of each component......')
    fragment = get_fragment(y) 
    print('Overlapping peak detection......')
    peaks = Peak_detection(fragment,point)

    for overpeak in range(len(peaks)):

        peak = peaks[overpeak]
        comindex = [fragment[j,0] for j in range(fragment.shape[0])]
        
        peak0=sorted([fragment[peak[j]][0] for j in range(len(peak))])
        peak1=sorted([fragment[peak[j]][1] for j in range(len(peak))])


        x = X[int(min(peak0)-range_point):int(max(peak1)+range_point+1),:]

        x, bias = back_remove(x,point,range_point)

        com = [];ind = []

        for r in range (len(peak)):
            ind.append(comindex.index(peak0[r]))
            com.append(components[ind[-1]])
        peak = ind
        
        
        if len(peak) <= 3:
            methodi = 'ITTFA'
            starttime = datetime.datetime.now()  
            if len(peak) == 1:
                ind_s = [np.argmax(np.abs(np.sum(x,1)))]

            elif len(peak) == 2:
                ch = np.abs(np.sum(x,1))
                ind_s1 = np.argmax(ch[int(peak0[0]-peak0[0]+range_point):int(peak0[1]-peak0[0]+range_point)])
                ind_s2 = np.argmax(ch[int(peak1[0]-peak0[0]+range_point):int(peak1[1]-peak0[0]+range_point)])
                ind_s = [int(ind_s1+range_point),int(ind_s2+peak1[0]-peak0[0]+range_point)]
            
            else:
                ch = np.abs(np.sum(x,1))
                ind_s1 = np.argmax(ch[int(peak0[0]-peak0[0]+range_point-point):int(peak0[1]-peak0[0]+range_point-point)])
                pm1 = int(max(peak1[0],peak0[1])-peak0[0]+range_point+point)
                pm2 = int(min(peak0[2],peak1[1])-peak0[0]+range_point-point)
                if pm1==pm2:
                    pm2=pm2+1
                ind_s2 = np.argmax(ch[min(pm1,pm2):max(pm1,pm2)])
                ind_s3 = np.argmax(ch[int(peak1[1]-peak0[0]+range_point+point):int(peak1[2]-peak0[0]+range_point+point)])
                ind_s = [int(ind_s1+range_point-point),int(ind_s2+min(pm1,pm2)),int(ind_s3+peak1[1]-peak0[0]+range_point+point)]

            c = np.zeros((x.shape[0], len(peak)))
            for i in range(len(peak)):  
                cc = ittfa(x, ind_s[i], len(peak))
                c[:, i] = cc[:,0]
            
            S = np.zeros((len(peak), x.shape[1]))
            for j in range(0, S.shape[1]):
                a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                S[:, j] = a['xx']

            re_chrom = np.zeros((len(peak), x.shape[0]))
            for k in range(len(peak)): 
                ck = c[:,k].reshape(x.shape[0],1)
                sk = S[k,:].reshape(1,x.shape[1])
                re_chrom[k,:]=np.sum(np.dot(ck,sk),1)


            R2 = explained_variance_score(x, np.dot(c,S), multioutput='variance_weighted')
            r2,p = pearsonr(np.sum(x,1),np.sum(re_chrom,0))
            print('ITTFA: R2=',R2,'r2=',r2)
            endtime = datetime.datetime.now()  
            print ('ITTFA: The time of ', str(len(peak)), '-component:',(endtime - starttime),".seconds")    
            
            plot_chrom(x,re_chrom,peak,methodi,result_path) 
            for num in range(len(peak)):
                component.append(com[num][0])
                overlapping_num.append(len(peak)) 
                retention_time.append(RT[int(np.argmax(re_chrom[num,:])+min(peak0)-range_point)])
                R_R2.append(R2)
                r_r2.append(r2)
                MCR_method.append(methodi)
                      
            if R2<0.98 and len(peak)==3:
                methodi = 'HELP_FR'
                starttime = datetime.datetime.now()  
                
                p = optim_frag(peak0,peak1,x.shape[0],point,range_point)
                R2_3 = []
            
                for opt in range (p.shape[0]):
                    re_chrom,R2 = mcr_by_fr(x,p,opt,peak,fragment)
                    R2_3.append(R2)
    
                optp = R2_3.index(max(R2_3))
                
                re_chrom,R2 = mcr_by_fr(x,p,optp,peak,fragment)
                r2,p = pearsonr(np.sum(x,1),np.sum(re_chrom,0))
                print('HELP_FR: R2=',R2,'r2=',r2)
            
                endtime = datetime.datetime.now()  
                print ('HELP_FR: The time of ', str(len(peak)), '-component:',(endtime - starttime),".seconds")

                plot_chrom(x,re_chrom,peak,methodi,result_path) 
                for num in range(len(peak)):
                    component.append(com[num][0])
                    overlapping_num.append(len(peak))
                    
                    retention_time.append(RT[int(np.argmax(re_chrom[num,:])+min(peak0)-range_point)])
                    R_R2.append(R2)
                    r_r2.append(r2)
                    MCR_method.append(methodi)
        else:
            print('Peak:'+str(overpeak)+': The overlapping peak contains more than three components.')
        print('--------peaks_',overpeak,'finished---------')   
        
    dataframe = pd.DataFrame({'component':component,
                              'retention_time/point':retention_time,
                              'overlapping_num':overlapping_num,
                              'R2':R_R2,'r2':r_r2,'MCR_method':MCR_method})

    dataframe.to_csv(str(result_path)+'/result.csv',index=False,sep=',')
  
if __name__ == '__main__':
    #Load the data file, model path and components information 
    channle = 3
    work_path = 'C:/Users/admin/Desktop/DeepResolution'
    
    data_file = work_path+'/data/zhi10-5vs1.CDF'
    ncr = netcdf_reader(data_file, bmmap=False)
    m = ncr.mat(1,3599, 1)
    RT = m['rt']
    Xdata = m['d']
    Xdata = Xdata[:,0:416]
        
    result_path = work_path+'/result/'+data_file.split('/')[-1]
    mkdir(result_path)
    
    component_file = csv.reader(open(work_path+'/data/component.csv', encoding='utf-8'))
    components = [row for row in component_file]

    Xtest = preprocess(Xdata,channle)
  
    # Set the root directory of models and reload the models one by one     
    ypred = np.zeros((33,Xtest.shape[0]))
    list_dirs = os.walk(work_path+'/model') 
    i=0
    starttime = datetime.datetime.now()  
    tf.compat.v1.reset_default_graph
    
    with tf.compat.v1.Session() as sess:
        new_saver=tf.compat.v1.train.import_meta_graph(work_path+'/model/component_01/compoent.ckpt.meta') 
        for root, dirs, files in list_dirs: 
            for d in dirs:
                os.chdir(os.path.join(root, d)) 
                new_saver.restore(sess,"./compoent.ckpt")
                graph = tf.compat.v1.get_default_graph()
                xs=graph.get_operation_by_name('xs').outputs[0]
                keep_prob=graph.get_operation_by_name('keep_prob').outputs[0]
     
                prediction = graph.get_tensor_by_name('prediction:0')
                test_ypred = sess.run(prediction,feed_dict={xs: Xtest, keep_prob : 1.0})  
                ypred[i,:] = test_ypred[:,0]
                i+=1
                
    endtime = datetime.datetime.now()  
    print ('The prediction time :',(endtime - starttime),".seconds")        

    RT_sta = np.searchsorted(RT,4.5)
    RT_end = np.searchsorted(RT,5)
    Hot_map(RT[RT_sta:RT_end],ypred[:,RT_sta:RT_end],result_path)
    MCR(components,Xdata,RT,ypred,result_path)
    
