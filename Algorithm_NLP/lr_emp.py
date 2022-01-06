import numpy as np
import sys
import matplotlib.pyplot as plt

def singleSGD(data,initials,eta):
    nobs=data.shape[0]
    theta=initials
    for i in np.arange(0,nobs,1):
        grad=1/nobs*(1/(1+np.exp((-1)*np.dot(theta,data[i,1:])))-data[i,0])*data[i,1:]
        theta=theta-eta*grad
    return(theta)

def NegL(X,theta,y):
    s=np.dot(X,theta).reshape((y.shape[0],1))
    y=y.reshape((y.shape[0],1))
    return(np.mean(np.multiply(s,y*(-1))+np.log(1+np.exp(s))))

def lr(formatted_train_input,formatted_validation_input,formatted_test_input, dict_input, train_out, test_out, metrics_out,num_epoch):
    formatted_train=np.loadtxt(
    fname=formatted_train_input,
    delimiter="\t",
    )

    
    formatted_validation=np.loadtxt(
    fname=formatted_validation_input,
    delimiter="\t",
    )
    
    formatted_test=np.loadtxt(
    fname=formatted_test_input,
    delimiter="\t",
    )
    train_nrows=formatted_train.shape[0]
    test_nrows=formatted_test.shape[0]
    valid_nrows=formatted_validation.shape[0]

    veclen=formatted_train.shape[1]
    formatted_train_dat=np.insert(formatted_train, 1, np.ones((train_nrows,)), axis=1)
    formatted_test_dat=np.insert(formatted_test, 1, np.ones((test_nrows,)), axis=1)
    formatted_validation_dat=np.insert(formatted_validation, 1, np.ones((valid_nrows,)), axis=1)

    init_r2=np.zeros(shape=(1,veclen))
    NegL_train_r2=[]
    NegL_valid_r2=[]
    for j in np.arange(0,int(num_epoch),1):
        theta_epo=singleSGD(formatted_train_dat,init_r2,0.1)
        NegL_train_r2.append(NegL(formatted_train_dat[:,1:],theta_epo.reshape((veclen,1)),formatted_train_dat[:,0]))
        NegL_valid_r2.append(NegL(formatted_validation_dat[:,1:],theta_epo.reshape((veclen,1)),formatted_validation_dat[:,0]))
        init_r2=theta_epo
    
    plt. clf()
    plt.plot(np.arange(0,int(num_epoch),1),NegL_train,"r--",label="train")
    plt.plot(np.arange(0,int(num_epoch),1),NegL_valid,"b--",label="valid")
    plt.legend()
    plt.savefig("Q1.4.1.pdf")

    plt. clf()
    plt.plot(np.arange(0,int(num_epoch),1),NegL_train_M2,"r--",label="train")
    plt.plot(np.arange(0,int(num_epoch),1),NegL_valid_M2,"b--",label="valid")
    plt.legend()
    plt.savefig("Q1.4.2.pdf")

    plt. clf()
    plt.plot(np.arange(0,int(num_epoch),1),NegL_train,"r--",label="alpha=0.01")
    plt.plot(np.arange(0,int(num_epoch),1),NegL_train_r1,"b--",label="alpha=0.001")
    plt.plot(np.arange(0,int(num_epoch),1),NegL_train_r2,"g--",label="alpha=0.1")
    plt.legend()
    plt.savefig("Q1.4.5.pdf")

    train_label=np.dot(formatted_train_dat[:,1:],init_M2.reshape((veclen,1)))
    for i in np.arange(0,train_nrows,1):
        if train_label[i,0]>=0:
            train_label[i,0]=1
        else:
            train_label[i,0]=0
    np.savetxt(train_out, train_label,fmt="%i")
    
    test_label=np.dot(formatted_test_dat[:,1:],init_M2.reshape((veclen,1)))
    for i in np.arange(0,test_nrows,1):
        if test_label[i,]>=0:
            test_label[i,]=1
        else:
            test_label[i,]=0
    
    np.savetxt(test_out, test_label,fmt="%i")

    validation_label=np.dot(formatted_validation_dat[:,1:],init.reshape((veclen,1)))
    for i in np.arange(0,valid_nrows,1):
        if validation_label[i,]>=0:
            validation_label[i,]=1
        else:
            validation_label[i,]=0
    
    
    trainTF,trainTF_counts=np.unique(np.array(train_label).reshape((train_nrows,1))== np.array(formatted_train[:,0]).reshape((train_nrows,1)),return_counts=True)
    if trainTF_counts[~trainTF].size==0:
        trainErr=0
    else:
        trainErr=trainTF_counts[~trainTF]/(train_nrows)

    testTF,testTF_counts=np.unique(np.array(test_label).reshape((test_nrows,1))== np.array(formatted_test[:,0]).reshape((test_nrows,1)),return_counts=True)
    if testTF_counts[~testTF].size==0:
        testErr=0
    else:
        testErr=testTF_counts[~testTF]/(test_nrows)
    
    with open(metrics_out, "w") as txt:
        print('error(train): %.6f' % trainErr, file=txt)
        print('error(test): %.6f' % testErr, file=txt)


if __name__ =='__main__':
    formatted_train_input=sys.argv[1]
    formatted_validation_input=sys.argv[2]
    formatted_test_input=sys.argv[3]
    dict_input=sys.argv[4]
    train_out=sys.argv[5]
    test_out=sys.argv[6]
    metrics_out=sys.argv[7]
    num_epoch=sys.argv[8]
    lr(formatted_train_input,formatted_validation_input,formatted_test_input, dict_input, train_out, test_out, metrics_out,num_epoch)