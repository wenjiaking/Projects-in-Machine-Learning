import numpy as np
import sys


def loaddata(path):
    with open(path,"r") as f:
        data = f.read().splitlines() 
        data = [i.split() for i in data]
    return(data)

def nploadtxt(path):
    data=np.loadtxt(fname=path, dtype="str", delimiter="\t")
    return(data)

def learnhmm (train_input,index_to_word,index_to_tag,hmminit,hmmemit,hmmtrans):
    train_list=loaddata(train_input)
    startInd=[i+1 for i, x in enumerate(train_list) if x==[]]
    startInd.insert(0,0)
    startInd_ext=[i+1 for i, x in enumerate(train_list) if x==[]]
    startInd_ext.insert(0,0)
    startInd_ext.append(len(train_list)+1)
    train_array=nploadtxt(train_input)
    wordIndex=nploadtxt(index_to_word)
    n_word=wordIndex.shape[0]
    tagIndex=nploadtxt(index_to_tag)
    n_tag=tagIndex.shape[0]
    train_init_tag=[train_list[i][1] for i in startInd]
    init=np.zeros((n_tag,1),dtype=float)
    emit=np.zeros((n_tag,n_word), dtype=float)
    trans=np.zeros((n_tag,n_tag),dtype=float)
    for i in np.arange(0,n_tag,1):
        init[i,0]=(train_init_tag.count(tagIndex[i])+1)/(len(train_init_tag)+n_tag)
        train_i=train_array[train_array[:,1]==tagIndex[i],:]
        words_i=list(train_i[:,0])
        emit[i,:]=[(words_i.count(wordIndex[j])+1)/(train_i.shape[0]+n_word) for j in np.arange(0,n_word,1)]
        tagi_counts=np.zeros((len(startInd),n_tag),dtype=float)
        for j in np.arange(0,len(startInd),1):
            trainlist_j=train_list[startInd[j]:(startInd_ext[j+1]-1)]
            seqtags_j=[w[1] for w in trainlist_j]
            seqtagsi_j=[seqtags_j[d+1] for d, x in enumerate(seqtags_j[:(len(seqtags_j)-1)]) if x==tagIndex[i]]
            tagi_counts[j,:]=[seqtagsi_j.count(tagIndex[t]) for t in np.arange(0,n_tag,1)]
        trans[i,]=(np.sum(tagi_counts,axis=0)+1)/(np.sum(tagi_counts)+n_tag)

    np.savetxt(hmminit, init) 
    np.savetxt(hmmemit, emit) 
    np.savetxt(hmmtrans, trans)   
    return(init,emit, trans)

    
#inittest,emittest,transtest=learnhmm("handout/toy_data/train.txt","handout/toy_data/index_to_word.txt","handout/toy_data/index_to_tag.txt","handout/toy_output/hmminit1.txt","handout/toy_output/hmmemit1.txt","handout/toy_output/hmmtrans1.txt")

if __name__ =='__main__':
    train_input=sys.argv[1]
    index_to_word=sys.argv[2]
    index_to_tag=sys.argv[3]
    hmminit=sys.argv[4]
    hmmemit=sys.argv[5]
    hmmtrans=sys.argv[6]
    learnhmm (train_input,index_to_word,index_to_tag,hmminit,hmmemit,hmmtrans)