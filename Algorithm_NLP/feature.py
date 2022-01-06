import numpy as np
import sys

def vocab(path, d):
    with open(path,"r") as f:
        data = f.read().splitlines() 
    
    nobs=len(data)
    nwords=len(d)
    vecInd = np.zeros(shape=(nobs,nwords+1))
    for i in np.arange(0,nobs,1):
        words=str(data[i]).split()
        vecInd[i,0]=words[0]
        for w in words[1:]:
            if w in d.keys():
                vecInd[i,int(d[w])+1]=1
    return(vecInd)

def w2vec(data,w2vd):
    nobs=data.shape[0]
    features=data[:,1]
    lenVec=list(w2vd.values())[0].shape[0]
    vecInd=np.zeros(shape=(nobs,lenVec))
    for i in np.arange(0,nobs,1):
        words=str(features[i,]).split()
        nwords=len(words)
        wordsInd = np.zeros(shape=(nwords,lenVec))
        m=0
        for j in np.arange(0,nwords,1):
            if words[j] in w2vd.keys():
                wordsInd[j,:]=w2vd[words[j]]
                m=m+1
        wordsVec=np.sum(wordsInd,axis = 0)/m
        vecInd[i,:]=wordsVec
    outVec=np.insert(vecInd, 0, data[:,0], axis=1)
    return(outVec)

def feature(train_input, validation_input,test_input,dict_input,formatted_train_out,formatted_validation_out,formatted_test_out,feature_flag,feature_dictionary_input):
    
    
    if (int(feature_flag)==1):
        d={}
        with open(dict_input) as dicf:
            for line in dicf:
                (key, val) = line.split()
                d[key] = int(val)
    
        train_out=vocab(train_input, d)
        test_out=vocab(test_input, d)
        validation_out=vocab(validation_input, d)
        np.savetxt(formatted_train_out,train_out,fmt='%i',delimiter='\t')
        np.savetxt(formatted_test_out,test_out,fmt='%i',delimiter='\t')
        np.savetxt(formatted_validation_out,validation_out,fmt='%i',delimiter='\t')
    else:
        train_dat=np.loadtxt(
            fname=train_input,
            delimiter="\t",
            dtype='str',
            )
        
        test_dat=np.loadtxt(
            fname=test_input,
            delimiter="\t",
            dtype='str',
            )
        
        validation_dat=np.loadtxt(
            fname=validation_input,
            delimiter="\t",
            dtype='str',
            )

        w2vd = {}
        with open(feature_dictionary_input) as w2vf:
            for line in w2vf:
                word = line.split()
                w2vd[word[0]] = np.float_(word[1:])
        
        train_out=w2vec(train_dat,w2vd)
        test_out=w2vec(test_dat, w2vd)
        validation_out=w2vec(validation_dat, w2vd)
        np.savetxt(formatted_train_out,train_out,fmt='%10.6f',delimiter='\t')
        np.savetxt(formatted_test_out,test_out,fmt='%10.6f',delimiter='\t')
        np.savetxt(formatted_validation_out,validation_out,fmt='%10.6f',delimiter='\t')

if __name__ =='__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input=sys.argv[4]
    formatted_train_out=sys.argv[5]
    formatted_validation_out=sys.argv[6]
    formatted_test_out=sys.argv[7]
    feature_flag=sys.argv[8]
    feature_dictionary_input=sys.argv[9]
    feature(train_input, validation_input,test_input,dict_input,formatted_train_out,formatted_validation_out,formatted_test_out,feature_flag,feature_dictionary_input)
    
    
                    
        
    