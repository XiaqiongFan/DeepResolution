# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: admin
"""


import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import datetime
import os
import random

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
      
def weights_variables(shape):
    weight = tf.Variable(tf.random.truncated_normal(shape,stddev=0.1))
    return weight
        
def bias_variables(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv1d(x,W):
    return tf.nn.conv1d(x,W,stride=2,padding='SAME') 
        
def max_pool(x):
    return tf.layers.max_pooling1d(x,pool_size=2,strides=2,padding='SAME')   
def randomize(dataset, labels):
    for i in range(5):
        permutation = np.random.permutation(labels.shape[0])
        dataset = dataset[permutation,:]
        labels = labels[permutation,:]
    return dataset, labels

def process(X,Y,channle):   
    m = int(X.shape[0])
    n = int(X.shape[1])
    r = int(X.shape[2])
    c = int(X.shape[3])
    q = int(channle*c)
    
    for i in range(m):
        for j in range(n):
            for k in range(r):
                X[i,j,k,:]=10000*X[i,j,k,:]/np.max(X[i,j,k,:])
    
    X_all = np.zeros((2,m,n,r,c))
    Y_all = np.zeros((2,m,n,r))
  
    X_r = np.zeros((X.shape))
    Y_r = np.zeros((Y.shape))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_r[i,j,:,:] = X[i,j,:,:][::-1]
            Y_r[i,j,:] = Y[i,j,:][::-1]

    X_all[0,:,:,:,:] = X;    Y_all[0,:,:,:]=Y
    X_all[1,:,:,:,:] = X_r;  Y_all[1,:,:,:]=Y_r
          
    size = int(channle/2)
    X_p = np.zeros((2,m,n,r,q))
    for u in range(2):
        for i in range(m):
            for j in range(n):
                for k in range(r):
                    
                    if k-size<0: 
                        Xkl = np.zeros((size,c))
                        Xk = np.concatenate((Xkl, X_all[u,i,j,k:k+size+1,:]), axis=0).reshape(1,q)
                        
                    elif k+size>=r:
                            
                        Xkr =  np.zeros((k+size+1-r,c))                         
                        Xk = np.concatenate((X_all[u,i,j,k-size:r,:], Xkr), axis=0).reshape(1,q)
                            
                    else:                          
                        Xk = X_all[u,i,j,k-size:k+size+1,:].reshape(1,q)
                            
                    X_p[u,i,j,k,:] = Xk                               

    X_data = X_p.reshape(int(2*m*n*r),q)           
        
    Y_1 = Y_all.reshape(int(2*m*n*r),1)   
    Y_2 = np.ones((Y_1.shape)) - Y_1
    Y_data = np.concatenate((Y_1,Y_2),axis=1)
    return X_data,Y_data

    
def Load_data(compound,channle,data_path):
    datafile1 = data_path+'/data_'+str(compound)+'.npy'
    X_raw = np.load(datafile1)
       
    datafile2 = data_path+'/labels_'+str(compound)+'.npy'
    Y_raw = np.load(datafile2) 

    X_data,Y_data = process(X_raw,Y_raw,channle)
   
    X,Y = randomize(X_data,Y_data)

    Xtrain = X[0:int(0.8*X.shape[0])]
    Xvalid = X[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
    Xtest = X[int(0.9*X.shape[0]):X.shape[0]]

    Ytrain = Y[0:int(0.8*X.shape[0])]
    Yvalid = Y[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
    Ytest = Y[int(0.9*X.shape[0]):X.shape[0]]
    return Xtrain,Xvalid,Xtest,Ytrain,Yvalid,Ytest
            
def train(compound,channle,data_path):
    Xtrain,Xvalid,Xtest,Ytrain,Yvalid,Ytest = Load_data(compound,channle,data_path)
    tf.compat.v1.reset_default_graph
    starttime = datetime.datetime.now()   
            
    xs= tf.compat.v1.placeholder(tf.float32,[None,416*channle],name='xs') 
    ys= tf.compat.v1.placeholder(tf.float32,[None,2])
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    x_image = tf.reshape(xs,[-1,416,channle])
    
    W_conv1 = weights_variables([2,channle,64])
    b_conv1 = bias_variables([64])
    h_conv1 = tf.nn.relu(conv1d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)    
             
    W_conv2 = weights_variables([2,64,128])
    b_conv2 = bias_variables([128])
    h_conv2 = tf.nn.relu(conv1d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)   

    size_conv = int(h_pool2.shape[1]*h_pool2.shape[2])
    W_fc1 = weights_variables([size_conv,1024])
    b_fc1 = bias_variables([1024])
    h_conv_flat = tf.reshape(h_pool2,[-1,size_conv])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat,W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
        
    W_fc2 = weights_variables([1024,2])
    b_fc2 = bias_variables([2])
    pred = tf.matmul(h_fc1_drop ,W_fc2) + b_fc2
    prediction = tf.nn.softmax(pred, name= 'prediction')
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = ys))    
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)    
    saver = tf.compat.v1.train.Saver()

    save_file = './model/component_'+str(compound)+'/compoent.ckpt'
    
    accuracy_valid = []
    loss_valid  = []
    batch_size = 100
    epochs = 50
    accuracy_valid.clear()
    loss_valid.clear()
    
    num_steps = Xtrain.shape[0] // batch_size
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(epochs):
            for step in range(num_steps):
                offset = (step * batch_size) % (Ytrain.shape[0] - batch_size)   
                batch_xs = Xtrain[offset:(offset + batch_size), :]
                batch_ys = Ytrain[offset:(offset + batch_size), :]
                feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5}
                _, loss, predictions = sess.run([train_step, cross_entropy, prediction], feed_dict=feed_dict)       
            
            feed_dict_val={xs:Xvalid,ys:Yvalid,keep_prob:1.0}       
            loss_val,pred_val = sess.run([cross_entropy,prediction], feed_dict=feed_dict_val)
            acc_val = accuracy(pred_val,Yvalid)
            print('The epoch', epoch, 'finished. The accuracy %.1f%%' % (acc_val),'loss =',loss_val)
            accuracy_valid.append(acc_val/100)
            loss_valid.append(loss_val)
        
        TIMES = [(i + 1) for i in range(0, epochs)]    
       
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(TIMES,accuracy_valid,'r',linewidth=2)
        ax1.set_ylabel('Accuracy',size=15,color='r')
        ax1.set_xlabel('Epoch',size=15)       
        ax2 = ax1.twinx()  
        ax2.plot(TIMES, loss_valid, 'g',linewidth=2)
        ax2.set_ylabel('Loss',size=15,color='g')
        ax = plt.gca()
        ax.spines['left'].set_color('r')
        ax.spines['right'].set_color('g')
        plt.title('compound_'+str(compound))

        saver.save(sess, save_file)
        print('Trained Model Saved.')
    
    endtime = datetime.datetime.now()  
    print ('The time :',(endtime - starttime),".seconds")
    
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        test_ypred = sess.run(prediction,feed_dict={ xs: Xtest, ys: Ytest, keep_prob : 1.0})    
    print('The test accuracy %.1f%%' % accuracy(test_ypred,Ytest))

if __name__ == '__main__':

    channle = 3
    data_path = u'./Data'
    
    compounds = 19
    print('Compound',compound,'start')

    train(compound,channle,data_path)

    print('Compound',compound,'finished')
