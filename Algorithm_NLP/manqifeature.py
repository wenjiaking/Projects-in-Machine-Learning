import numpy as np
import sys
import time
#import os
#os.chdir("/Users/littleveg/Desktop/10601/hw4/test")

#test_start = time.time()

def loaddata(path):
    with open(path,"r") as f:
        data = f.read().splitlines() 
        data = [i.split() for i in data]
    return(data)

def model1_feature(data):
    formatted_dat = []
    for ele in data:
        tmp = np.zeros(len(mydict)+1, dtype=int)
        tmp[0]=int(ele[0])
        for word in ele:
            if word in mydict:
                tmp[int(mydict[word])+1] = 1
        tmp = [str(format(i)) for i in tmp]
        tmp = "\t".join(tmp)
        formatted_dat.append(tmp)
    return(formatted_dat)
    
def model2(data):

    formatted_dat2  = []
    for ele in data:
        tmp = []
        for word in ele:
            if word in word2vec_dict:
                tmp.append(word2vec_dict[word])
        tmp = np.sum(tmp,axis = 0 )/len(tmp)
        tmp = [str(format(round(float(i),6), '.6f')) for i in tmp]
        tmp = "\t".join(tmp)
        out = str(format(float(ele[0]), '.6f'))+'\t'+tmp
        formatted_dat2.append(out)
    return(formatted_dat2)



train_input = sys.argv[1]
valid_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
train_out = sys.argv[5]
valid_out = sys.argv[6]
test_out = sys.argv[7]
feature_flag = int(sys.argv[8])
ftdict_input = sys.argv[9]


# train_input = "smalldata/train_data.tsv"
# valid_input = "smalldata/valid_data.tsv"
# test_input = "smalldata/test_data.tsv"
# dict_input = "dict.txt"
# train_out = "smalloutput/formatted_train.tsv"
# valid_out = "smalloutput/formatted_valid.tsv"
# test_out = "smalloutput/formatted_test.tsv"
# feature_flag = 1
# ftdict_input = "word2vec.txt"

train_dat = loaddata(train_input)

valid_dat = loaddata(valid_input)

test_dat = loaddata(test_input)

if feature_flag == 1:
    with open(dict_input,"r") as f:
        mydict = dict(i.split() for i in f)
    
    formattd_train = model1_feature(train_dat)
    formattd_valid = model1_feature(valid_dat)
    formattd_test = model1_feature(test_dat)
elif feature_flag == 2:

    with open(ftdict_input,"r") as f:
        word2vec = f.read().splitlines() 
        word2vec_dict = {}
        for j in range(len(word2vec)):
            tmp = word2vec[j].split()
            word2vec_dict[tmp[0]] = [float(i) for i in tmp[1:]]

    formattd_train = model2(train_dat)
    formattd_valid = model2(valid_dat)
    formattd_test = model2(test_dat)

with open(train_out,'w') as f:
    for l in formattd_train:
      f.write(l+'\n') 

with open(valid_out,'w') as f:
    for l in formattd_valid:
      f.write(l+'\n') 

with open(test_out,'w') as f:
    for l in formattd_test:
      f.write(l+'\n') 

#print("--- %s seconds ---" % (time.time() - test_start))
#print("stop")

# start_time = time.time
# formattd_train = model1_feature(train_dat)

# test = model2(train_dat)
# print("--- %s seconds ---" % (time.time() - start_time))


# tmp = []
# for word in train_dat[0]:
#     if word in word2vec_dict:
#         tmp.append(word2vec_dict[word])
# tmp = np.sum(tmp,axis = 0 )/len(tmp)
# tmp = [round(float(i),6) for i in tmp]
# out = np.ones(len(tmp)+1, dtype=float)
# out[0]=float(train_dat[0][0])
# out[1:] = tmp
# tmp = [str(element) for element in tmp]
# tmp = "\t".join(tmp)