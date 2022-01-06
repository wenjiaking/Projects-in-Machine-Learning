import numpy as np
import sys


def NNFORWARD(x,y,alpha,beta):
    a=np.dot(alpha,x)
    z=1/(1+np.exp(-a))
    z_ext=np.insert(z, 0, 1)
    b=np.dot(beta,z_ext)
    yhat=np.exp(b)/np.sum(np.exp(b))
    J=(-1)*np.log(yhat[int(y)])
    return(a,z,z_ext,b,yhat,J)

def NNBACKWARD(x,y,beta,yhat,z_ext,z):
    gb=yhat
    gb[int(y)]=yhat[int(y)]-1
    gz=np.dot((beta[:,1:]).T, gb)
    gbeta=np.outer(gb, z_ext)
    ga=gz * z * (1-z)
    galpha=np.outer(ga, x)
    return(galpha,gbeta)

def crossentropy(sample, alpha, beta):
    X=sample[:,1:]
    y=sample[:,0]
    A=np.dot(alpha,X.T)
    Z=1/(1+np.exp(-A))
    Z=np.insert(Z,0,np.ones((Z.shape[1],)),axis=0)
    B=np.dot(beta,Z)
    B_sum=np.sum(np.exp(B),axis=0)
    Yhat=np.exp(B)/B_sum[None,:]
    Y=np.zeros((4,y.shape[0]))
    for i in range(y.shape[0]):
        Y[int(y[i]),i]=1
    ce_J= np.sum(-Y*np.log(Yhat),axis=0)
    return(sum(ce_J)/y.shape[0])

def predict(sample, alpha, beta,out):
    X=sample[:,1:]
    y=sample[:,0]
    A=np.dot(alpha,X.T)
    Z=1/(1+np.exp(-A))
    Z=np.insert(Z,0,np.ones((Z.shape[1],)),axis=0)
    B=np.dot(beta,Z)
    B_sum=np.sum(np.exp(B),axis=0)
    Yhat=np.exp(B)/B_sum[None,:]
    Ylab=np.argmax(Yhat.T,axis=1)
    TF,TF_counts=np.unique(np.array([int(i) for i in y])==Ylab,return_counts=True)
    if TF_counts[~TF].size==0:
        e=0
    else:
        e=TF_counts[~TF]/y.shape[0]

    with open(out, "a") as lab_file:
        for i in Ylab:
            lab_file.write(('{}\n'.format(i)))

    return(e[0])



def SGD(train_input,valid_input,num_epoch,hidden_units,init_flag,learning_rate):
    train_data= np.loadtxt(train_input, dtype = 'float', delimiter=',')
    valid_data= np.loadtxt(valid_input, dtype = 'float',delimiter=",")

    num_epoch=int(num_epoch)
    hidden_units=int(hidden_units)
    init_flag=int(init_flag)
    learning_rate=float(learning_rate)

    nobs,nfeat=train_data.shape
    nobs_valid,nfeat_valid=valid_data.shape
    salpha=np.zeros((hidden_units,nfeat), dtype=float)
    sbeta=np.zeros((4,hidden_units+1), dtype=float)
    if init_flag==2:
        alpha=np.zeros((hidden_units,nfeat), dtype=float)
        beta=np.zeros((4,hidden_units+1), dtype=float)
    else:
        alpha=np.random.uniform(low=-0.1,high=0.1,size=(hidden_units,nfeat-1))
        alpha=np.insert(alpha, 0,np.zeros((hidden_units,)) , axis=1)
        beta=np.random.uniform(low=-0.1,high=0.1,size=(4,hidden_units))
        beta=np.insert(beta, 0,np.zeros((4,)) , axis=1)
    
    train_ext=np.insert(train_data, 1,np.ones((nobs,)) , axis=1)
    valid_ext=np.insert(valid_data, 1,np.ones((nobs_valid,)) , axis=1)
    ce_train=list()
    ce_valid=list()
    eps=1e-5
    for i in np.arange(0,num_epoch,1):
        for j in np.arange(0,nobs,1):
            x=train_ext[j,1:]
            y=train_ext[j,0]
            a,z,z_ext,b,yhat,J=NNFORWARD(x,y,alpha,beta)
            galpha,gbeta=NNBACKWARD(x,y,beta,yhat,z_ext,z)
            salpha=salpha+galpha*galpha
            sbeta=sbeta+gbeta*gbeta
            alpha=alpha-learning_rate*galpha/np.sqrt(salpha+eps)
            print(alpha)
            beta=beta-learning_rate*gbeta/np.sqrt(sbeta+eps)
            print(beta)
        ce_train.append(crossentropy(train_ext, alpha, beta))
        ce_valid.append(crossentropy(valid_ext, alpha, beta))
    
    entropy_train=crossentropy(train_ext, alpha, beta)
    entropy_valid=crossentropy(valid_ext, alpha, beta)
    return(ce_train,ce_valid)
    #return(entropy_train,entropy_valid)

list_train=[]
list_val=[]
for n in [5,20,50,100,200]:
    entropy_train,entropy_valid,ce_train,ce_valid=SGD("data/small_train.csv","data/small_val.csv",100,n,1,0.01)
    list_train.append(entropy_train)
    list_val.append(entropy_valid)

import matplotlib.pyplot as plt
plt. clf()
plt.plot([5,20,50,100,200],list_train,"r--",label="train")
plt.plot([5,20,50,100,200],list_val,"b--",label="valid")
plt.legend()
plt.savefig("Q3.1a.pdf")

los_ada_train,los_ada_valid=SGD("data/small_train.csv","data/small_val.csv",100,50,1,0.01)   
los_sgd_valid=np.loadtxt("val_loss_sgd_out.txt", dtype = 'float')
plt. clf()
plt.plot(np.arange(1,101),los_ada_valid,"r--",label="SGD with Adagrad")
plt.plot(np.arange(1,101),los_sgd_valid,"b--",label="SGD")
plt.legend()
plt.savefig("Q3.1c.pdf")

loss1_train,loss1_valid=SGD("data/small_train.csv","data/small_val.csv",100,50,1,0.1) 
loss2_train,loss2_valid=SGD("data/small_train.csv","data/small_val.csv",100,50,1,0.01) 
loss3_train,loss3_valid=SGD("data/small_train.csv","data/small_val.csv",100,50,1,0.001) 

plt. clf()
plt.plot(np.arange(1,101),loss1_train,"r--",label="train_LR0.1")
plt.plot(np.arange(1,101),loss1_valid,"b--",label="valid_LR0.1")
plt.legend()
plt.savefig("Q3.2a1.pdf")

plt. clf()
plt.plot(np.arange(1,101),loss2_train,"r--",label="train_LR0.01")
plt.plot(np.arange(1,101),loss2_valid,"b--",label="valid_LR0.01")
plt.legend()
plt.savefig("Q3.2a2.pdf")

plt. clf()
plt.plot(np.arange(1,101),loss3_train,"r--",label="train_LR0.001")
plt.plot(np.arange(1,101),loss3_valid,"b--",label="valid_LR0.001")
plt.legend()
plt.savefig("Q3.2a3.pdf")

if __name__ =='__main__':
    train_input=sys.argv[1]
    valid_input=sys.argv[2]
    train_out=sys.argv[3]
    valid_out=sys.argv[4]
    metrics_out=sys.argv[5]
    num_epoch=sys.argv[6]
    hidden_units=sys.argv[7]
    init_flag=sys.argv[8]
    learning_rate=sys.argv[9]
    SGD(train_input,valid_input,train_out,valid_out,metrics_out,num_epoch,hidden_units,init_flag,learning_rate)




