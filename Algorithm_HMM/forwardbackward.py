import numpy as np
import sys

def alphabeta(init,emit,trans,words,wordIndex):
    n_word=len(words)
    n_tag=trans.shape[0]
    log_emit=np.log(emit)
    log_trans=np.log(trans)

    alpha=np.zeros((n_tag,n_word),dtype=np.longdouble)
    beta=np.zeros((n_tag,n_word),dtype=np.longdouble)
    w0ind=wordIndex.get(words[0])
    alpha[:,0]=np.log(init)+log_emit[:,w0ind]
    for t in np.arange(1,n_word,1):
        wind=wordIndex.get(words[t])
        alpha_temp=np.exp(alpha[:,t-1].reshape((n_tag,1))+log_trans)
        alpha[:,t]=log_emit[:,wind]+np.log(np.sum(alpha_temp,0))
        wind_b=wordIndex.get(words[n_word-t])
        beta_temp=np.exp(log_emit[:,wind_b].reshape((n_tag,1))+beta[:,n_word-t].reshape((n_tag,1))+log_trans.T)
        beta[:,(n_word-1-t)]=np.log(np.sum(beta_temp,axis=0))
    
    return(alpha,beta)

def loaddata(path):
    with open(path,"r") as f:
        data = f.read().splitlines() 
        data = [i.split() for i in data]
    return(data)

def nploadtxt(path):
    data=np.loadtxt(fname=path, dtype="str", delimiter="\t")
    return(data)

def index_dict(indexfile):
    indexdict = {}
    i  = 0
    with open(indexfile, 'r') as f:
        lines = f.readlines()
    for l in lines:
        pairs = l.strip().split('\t')
        key = pairs[0]
        val = i
        indexdict[key]=val
        i=i+1
    return indexdict

def forwardbackward (validation_input,index_to_word,index_to_tag,hmminit,hmmemit,hmmtrans,predicted_file,metrics_file):
    valid_list=loaddata(validation_input)
    wordIndex=index_dict(index_to_word)
    tags=np.genfromtxt(index_to_tag, dtype=str)
    init=np.loadtxt(fname=hmminit,dtype=float)
    emit=np.loadtxt(fname=hmmemit,dtype=float)
    trans=np.loadtxt(fname=hmmtrans,dtype=float)
    startInd=[i+1 for i, x in enumerate(valid_list) if x==[]]
    startInd.insert(0,0)
    startInd_ext=[i+1 for i, x in enumerate(valid_list) if x==[]]
    startInd_ext.insert(0,0)
    startInd_ext.append(len(valid_list)+1)
    l_total=0
    error_counts=0
    words_total=0

    txt = open(predicted_file, "w")
    txt.write("")
    txt.close()

        
    for j in np.arange(0,len(startInd),1):
            wordlist=valid_list[startInd_ext[j]:(startInd_ext[j+1]-1)]
            words_j=[x[0] for x in wordlist]
            alpha_j,beta_j=alphabeta(init,emit,trans,words_j,wordIndex)
            ypred_j=tags[np.argmax(alpha_j+beta_j,axis=0)]
            tags_j=[x[1] for x in wordlist]
            error_counts += np.sum( ypred_j!=tags_j)
            words_total += len(words_j)

            l_j=np.log(np.sum(np.exp(alpha_j[:,-1])))
            l_total += l_j

            word_tag_j = np.hstack((np.array(words_j,dtype="str").reshape((len(words_j),1)), ypred_j.reshape((len(words_j),1))))
            if (j<(len(startInd)-1)):
                with open (predicted_file,'a') as txt:
                    np.savetxt(txt, word_tag_j, delimiter='\t', fmt="%s")
                    txt.write("\n")
                
            else:
                with open (predicted_file,'a') as txt:
                    np.savetxt(txt, word_tag_j, delimiter='\t', fmt="%s")

    
    l_avg = l_total/len(startInd)
    accuracy = 1 - (error_counts/words_total)

    txt=open(metrics_file, "w")
    txt.write("Average Log-Likelihood: "+ str(l_avg) + "\n")
    txt.write("Accuracy: " + str(accuracy))
    txt.close()

    return(l_avg,accuracy)

#average_l,average_accuracy=forwardbackward (validation_input,index_to_word,index_to_tag,hmminit,hmmemit,hmmtrans,predicted_file,metrics_file)

if __name__ =='__main__':
    validation_input=sys.argv[1]
    index_to_word=sys.argv[2]
    index_to_tag=sys.argv[3]
    hmminit=sys.argv[4]
    hmmemit=sys.argv[5]
    hmmtrans=sys.argv[6]
    predicted_file=sys.argv[7]
    metrics_file=sys.argv[8]
    forwardbackward (validation_input,index_to_word,index_to_tag,hmminit,hmmemit,hmmtrans,predicted_file,metrics_file)