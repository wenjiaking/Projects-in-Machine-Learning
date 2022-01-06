import numpy as np
import sys

def singleSGD(data,initials,eta):
    nobs=data.shape[0]
    theta=initials
    for i in np.arange(0,nobs,1):
        grad=1/nobs*(1/(1+np.exp((-1)*np.dot(theta,data[i,1:])))-data[i,0])*data[i,1:]
        theta=theta-eta*grad
    return(theta)

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

    init=np.zeros(shape=(1,veclen))
    for j in np.arange(0,int(num_epoch),1):
        theta_epo=singleSGD(formatted_train_dat,init,0.01)
        init=theta_epo
        
    train_label=np.dot(formatted_train_dat[:,1:],init.reshape((veclen,1)))
    for i in np.arange(0,train_nrows,1):
        if train_label[i,0]>=0:
            train_label[i,0]=1
        else:
            train_label[i,0]=0
    np.savetxt(train_out, train_label,fmt="%i")
    
    test_label=np.dot(formatted_test_dat[:,1:],init.reshape((veclen,1)))
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