import sys
import numpy as np
from numpy.lib.function_base import delete

class Node:
    def __init__(self,key):
        self.left=None
        self.right=None
        self.depth=key
        self.data=None
        self.attrused={}
        self.label=None


def trainTree (node,maxdepth):
    if maxdepth==0:
        dat=np.delete(node.data,0,0)
        nr_all,nc_all=dat.shape
        label_all,freq_all=np.unique(dat[:,nc_all-1],return_counts=True)
        node.label=label_all[np.argmax(freq_all)]
        return(node)
    else:
        dat=np.delete(node.data,0,0)
        nr_temp,nc_temp=dat.shape
        label_temp,freq_temp=np.unique(dat[:,nc_temp-1],return_counts=True)
        if node.depth>=maxdepth or nc_temp==1 or freq_temp[0]==nr_temp:
            if freq_temp[0]==nr_temp:
                node.label=label_temp[0]
            else:
                if freq_temp[0]==freq_temp[1]:
                    node.label=sorted(label_temp,reverse=True)[0]
                else:
                    node.label=label_temp[np.argmax(freq_temp)]

            
            #print("attach maxdepth or run out of attribute or pure data!")
            return(node)
        else:
            attri_best=bestfinder(node)
            if attri_best==None:
                node.label=label_temp[np.argmax(freq_temp)]
                return(node)
                #print("no more contributing attribute!")
            else:
                attri_index=np.where(node.data[0,:]==attri_best)[0]
                attri_value=np.unique(dat[:,attri_index])

                left_dat=dat[np.where(dat[:,attri_index]==attri_value[0])[0],:]
                left_dat=np.vstack((node.data[0,:],left_dat))
                left_dat=np.delete(left_dat,attri_index,1)
                

                right_dat=dat[np.where(dat[:,attri_index]==attri_value[1])[0],:]
                right_dat=np.vstack((node.data[0,:],right_dat))
                right_dat=np.delete(right_dat,attri_index,1)
                

                node.attrused[attri_best]=np.array([attri_value])
                node.depth=node.depth+1
                node.left=Node(node.depth)
                node.right=Node(node.depth)
                node.left.data=left_dat
                node.right.data=right_dat
                node.left=trainTree (node.left,maxdepth)
                node.right=trainTree (node.right,maxdepth)
                
                return(node)
            

            

def helper(node,col):
    dat=np.delete(node.data,0,0)
    nrows,ncols=dat.shape
    label, freq=np.unique(dat[:,ncols-1],return_counts=True)
    entropy=-(freq[1]/nrows)*np.log2(freq[1]/nrows)-(1-freq[1]/nrows)*np.log2(1-freq[1]/nrows)
    attri_class=np.unique(dat[:,col])
    if attri_class.shape[0]==1:
        c_entropy=entropy
    else:
        subdata0=dat[(dat[:,col]==attri_class[0]),:]
        subdata1=dat[(dat[:,col]==attri_class[1]),:]
        nrows0,ncols0=subdata0.shape
        nrows1,ncols1=subdata1.shape
        labels_subdata0,freq_subdata0=np.unique(subdata0[:,ncols0-1],return_counts=True)
        labels_subdata1,freq_subdata1=np.unique(subdata1[:,ncols1-1],return_counts=True)
        if freq_subdata1[0]==nrows1 and freq_subdata0[0]!=nrows0:
            c_entropy=-nrows0/nrows*(freq_subdata0[0]/nrows0*np.log2(freq_subdata0[0]/nrows0)+freq_subdata0[1]/nrows0*np.log2(freq_subdata0[1]/nrows0))
        else:
            if freq_subdata0[0]==nrows0 and freq_subdata1[0]!=nrows1:
                c_entropy=-nrows1/nrows*(freq_subdata1[0]/nrows1*np.log2(freq_subdata1[0]/nrows1)+freq_subdata1[1]/nrows1*np.log2(freq_subdata1[1]/nrows1))
            else:
                if freq_subdata0[0]==nrows0 and freq_subdata1[0]==nrows1:
                    c_entropy=0
                else:
                    c_entropy=-nrows0/nrows*(freq_subdata0[0]/nrows0*np.log2(freq_subdata0[0]/nrows0)+freq_subdata0[1]/nrows0*np.log2(freq_subdata0[1]/nrows0))-nrows1/nrows*(freq_subdata1[0]/nrows1*np.log2(freq_subdata1[0]/nrows1)+freq_subdata1[1]/nrows1*np.log2(freq_subdata1[1]/nrows1))
    
    mutualinfo=entropy-c_entropy
    return(np.array([entropy,mutualinfo]))



def bestfinder(node):
    nr,nc=node.data.shape
    MI0=0
    attri_best=None
    for value in np.array(range(nc-1)):
        MI=helper(node,value)
        if MI[1]>MI0:
            MI0=MI[1]
            attri_best=node.data[0,value]
    
    return(attri_best)

def printTree (Tree):
    if Tree.attrused!={}:
        temp_depth=Tree.depth
        temp_attr_name=list(Tree.attrused.keys())[0]
        temp_attr_values=list(Tree.attrused.values())[0]
        temp_left_dat=np.delete(Tree.left.data,0,0)
        left_ncol=temp_left_dat.shape[1]
        left_label,left_freq=np.unique(temp_left_dat[:,left_ncol-1],return_counts=True)
        
        left_str=list()
        left_str.extend(np.repeat("|",temp_depth))
        left_str.append(temp_attr_name)
        left_str.append("=") 
        left_str.append(temp_attr_values[0,0])
        left_str.append(":")
        left_str.extend(left_label)
        left_str.append("@")
        left_str.extend(list(map(str,left_freq)))
        str_sep=" "
        print(str_sep.join(left_str))
        printTree(Tree.left)
       
        temp_right_dat=np.delete(Tree.right.data,0,0)
        right_ncol=temp_right_dat.shape[1]
        right_label,right_freq=np.unique(temp_right_dat[:,right_ncol-1],return_counts=True)

        right_str=list()
        right_str.extend(np.repeat("|",temp_depth))
        right_str.append(temp_attr_name)
        right_str.append("=") 
        right_str.append(temp_attr_values[0,1])
        right_str.append(":")
        right_str.extend(right_label)
        right_str.append("@")
        right_str.extend(list(map(str,right_freq)))
        print(str_sep.join(right_str))
        printTree(Tree.right)


def predictTree(node, newdata):
    attr_select=node.attrused
    if attr_select=={}:
        return(node.label)
    else:
        attr_name=list(node.attrused.keys())[0]
        attr_values=list(node.attrused.values())[0]
        if newdata[1,newdata[0,:]==attr_name]==attr_values[0,0]:
            return(predictTree(node.left,newdata))
        else:
            return(predictTree(node.right,newdata))



def decisionTree(train_input, test_input,maxdepth,train_out,test_out,metrics_out):
    maxdepth=int(maxdepth)
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
    nr_train, nc_train=train_dat.shape
    nr_test,nc_test=test_dat.shape
    root=Node(0)
    root.data=train_dat
    Tree=trainTree (root,maxdepth)

    root_str=list()
    root_label,root_freq=np.unique(np.delete(train_dat,0,0)[:,nc_train-1],return_counts=True)
    root_str.extend(root_label)
    root_str.append("@")
    root_str.extend(list(map(str,root_freq)))
    str_sep=" "
    print(str_sep.join(root_str))
    printTree (Tree)

    train_labels=list()
    for i in np.arange(1,nr_train,1):
        newdata=np.vstack((train_dat[0,:],train_dat[i,:]))
        train_label=predictTree(Tree, newdata)
        train_labels.append(train_label)
    np.savetxt(train_out, train_labels,fmt="%s")

    test_labels=list()
    for j in np.arange(1,nr_test,1):
        newdata=np.vstack((test_dat[0,:],test_dat[j,:]))
        test_label=predictTree(Tree, newdata)
        test_labels.append(test_label)
    
    np.savetxt(test_out, test_labels,fmt="%s")
    trainTF,trainTF_counts=np.unique(np.array(train_labels).reshape((nr_train-1,1))==np.array(train_dat[np.arange(1,nr_train,1),nc_train-1]).reshape((nr_train-1,1)),return_counts=True)
    trainErr=trainTF_counts[~trainTF]/(nr_train-1)
    
    testTF,testTF_counts=np.unique(np.array(test_labels).reshape((nr_test-1,1))==np.array(test_dat[np.arange(1,nr_test,1),nc_test-1]).reshape((nr_test-1,1)),return_counts=True)
    testErr=testTF_counts[~testTF]/(nr_test-1)
    
    with open(metrics_out, "w") as txt:
        print('error(train): %.6f' % trainErr, file=txt)
        print('error(test): %.6f' % testErr, file=txt)

if __name__ =='__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    maxdepth=sys.argv[3]
    train_out=sys.argv[4]
    test_out=sys.argv[5]
    metrics_out=sys.argv[6]
    print("path to the training input: %s" % (train_input)) 
    print("path to the test input: %s" % (test_input)) 
    print("maximum depth to which the tree should be built: %s" % (maxdepth))
    print("predictions on the training data: %s" % (train_out))
    print("predictions on the test data: %s" % (test_out))
    print("train and test error: %s" % (metrics_out))
    decisionTree(train_input, test_input,maxdepth,train_out,test_out,metrics_out)
    
#decisionTree("education_train.tsv", "education_test.tsv",3,"education_train.labels","education_test.labels","education_matrics.txt")