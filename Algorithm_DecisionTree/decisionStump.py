import sys
import numpy as np


def decisionStump(train_input, test_input, split_index, train_out, test_out,metrics_out):
    split_index=int(split_index)
    train_array = np.loadtxt(
    fname=train_input,
    delimiter="\t",
    skiprows=1,
    dtype='str',
)
    test_array = np.loadtxt(
    fname=test_input,
    delimiter="\t",
    skiprows=1,
    dtype='str',
)
    
    train_nrows, train_ncols =train_array.shape
    test_nrows, test_ncols =test_array.shape
    
    attri_class=np.unique(train_array[:,split_index])
    label_element0, label_count0=np.unique(train_array[train_array[:,split_index]==attri_class[0],train_ncols-1],return_counts=True)
    label_element1, label_count1=np.unique(train_array[train_array[:,split_index]==attri_class[1],train_ncols-1],return_counts=True)
    vote_class0=label_element0[np.argmax(label_count0)]
    vote_class1=label_element1[np.argmax(label_count1)]
    
    train_labels=np.array(train_array[:,train_ncols-1])
    train_labels[train_array[:,split_index]==attri_class[0]]=vote_class0
    train_labels[train_array[:,split_index]==attri_class[1]]=vote_class1
    np.savetxt(train_out, train_labels,fmt="%s")
    
    test_labels=np.array(test_array[:,test_ncols-1])
    test_labels[test_array[:,split_index]==attri_class[0]]=vote_class0
    test_labels[test_array[:,split_index]==attri_class[1]]=vote_class1
    np.savetxt(test_out, test_labels,fmt="%s")
    
    trainTF,trainTF_counts=np.unique(np.array(train_labels).reshape((train_nrows,1))==np.array(train_array[:,train_ncols-1]).reshape((train_nrows,1)),return_counts=True)
    trainErr=trainTF_counts[~trainTF]/train_nrows
    
    testTF,testTF_counts=np.unique(np.array(test_labels).reshape((test_nrows,1))==np.array(test_array[:,test_ncols-1]).reshape((test_nrows,1)),return_counts=True)
    testErr=testTF_counts[~testTF]/test_nrows
    
    with open(metrics_out, "w") as txt:
        print('error(train): %.6f' % trainErr, file=txt)
        print('error(test): %.6f' % testErr, file=txt)
        
if __name__ =='__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index=sys.argv[3]
    train_out=sys.argv[4]
    test_out=sys.argv[5]
    metrics_out=sys.argv[6]
    print("path to the training input: %s" % (train_input)) 
    print("path to the test input: %s" % (test_input)) 
    print("the index of feature at which we split the dataset: %s" % (split_index))
    print("predictions on the training data: %s" % (train_out))
    print("predictions on the test data: %s" % (test_out))
    print("train and test error: %s" % (metrics_out))
    decisionStump(train_input, test_input, split_index, train_out, test_out,metrics_out)
    
    
    
    
    
   

    
    