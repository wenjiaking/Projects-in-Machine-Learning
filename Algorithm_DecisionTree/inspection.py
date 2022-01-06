import sys
import numpy as np

def inspection(inputfile, outputfile):
    data_array = np.loadtxt(
    fname=inputfile,
    delimiter="\t",
    skiprows=1,
    dtype='str',
)
    nrows, ncols =data_array.shape
    label, freq=np.unique(data_array[:,ncols-1],return_counts=True)
    entropy=-(freq[1]/nrows)*np.log2(freq[1]/nrows)-(1-freq[1]/nrows)*np.log2(1-freq[1]/nrows)
    errorrate=np.min(freq)/nrows
    with open(outputfile, "w") as txt:
        print('entropy: %.6f' % entropy, file=txt)
        print('error: %.6f' % errorrate, file=txt)


if __name__ == ’__main__’:
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    print("path to the input data: %s" % (inputfile)) 
    print("path to the entropy and error rate: %s" % (metrics_out))
    inspection(inputfile, outputfile)