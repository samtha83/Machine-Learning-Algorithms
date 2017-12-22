import sys
import math
import random

from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score


#---Reading the datafile---#
datafile= sys.argv[1]
f = open(datafile,'r')
#f = open("datafile.txt")

data = []
i=0;
l= f.readline()
while(l!=''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(a[j])      
    data.append(l2)
    #data[i].append(1)
    i=i+1;
    l = f.readline()
    
rows = len(data)
cols = len(data[0])

#print("row=", rows, "cols= ", cols)
#for i in range(0,rows,1):
#    print(data[i])



#---Reading the trainlabelfile-----
trainlabelfile = sys.argv[2]
f = open(trainlabelfile,'r')
#f = open("trainlabelfile.txt")

trainlabels={} #Initializing the dictionary
i=0;
l= f.readline()
   
while(l!=''):
    a = l.split()
    if(int(a[0])==0):# storing key : value in dictionary
        trainlabels[int(a[1])]=int(a[0])-1 #storing 0 the label as -1
    else:
        trainlabels[int(a[1])]=int(a[0])
    l = f.readline()

#---extracting just labels------------------------
labellist=[]
for label in trainlabels:
    labellist.append(trainlabels.get(label))


#------------calculating dot product-----------------

def dot_product(w,data):
    sum_dp=0
    for j in range(0,cols,1):
        sum_dp = sum_dp+ w[j]*float(data[j])
    return sum_dp


#-----------logic--------------------------------------------


import math
newdata_train =[]
#newdata_test=[]
filename= sys.argv[3]
f = open(filename,'w')
#f = open("results.txt","w")


planes=[10, 100, 1000,10000]

for k in planes:
    print("For ",k," random planes")
    for i  in range(0,k,1):
        
    #l=l+1
    
        list_train=[]
    #list_test=[]
    
    #for i in range(0, rows, 1):
        #if(trainlabels.get(i)!=None):
            #list_train.append(0)
        #else:
            #list_test.append(0)
    
        w=[]
        for j in range(0, cols, 1):
            w.append(0)

        for j in range(0, cols, 1):
            w[j]=w[j] + random.uniform(1,-1)
    
        for i in range(0,rows,1):
            dp=0
            dp=dot_product(w,data[i])
            sign=int(math.copysign(1, dp))
            val=int((1+sign)/2)
        #print("sign of dp:",sign)
        #if(trainlabels.get(i)!=None):
            list_train.append(val)
            #c=c+1
        #else:
            #list_test.append(sign)
            #d=d+1
    
    
        newdata_train.append(list_train)
    #newdata_test.append(list_test)
    
        newdata_train_t=zip(*newdata_train)
        traindata = []
        for row in newdata_train_t:
            traindata.append(row)

    clf = svm.SVC(kernel='linear', C=.01)
    scores = cross_val_score(clf, traindata, labellist, cv=5)
    scores[:]= [1-x for x in scores]
    scores_o = cross_val_score(clf, data, labellist, cv=5)
    scores_o[:]=[1-x for x in scores_o]
    
    print("error for new features data: ",scores)
    print("mean error for new features data: ",scores.mean())
    print("error for orginal data: ",scores_o)
    print("mean error for orginal data: ",scores_o.mean())



    f.write("error for new features data: "+ "\n")
    f.write(str(scores)+"\n")
    f.write("mean error for new features data: "+ "\n")
    f.write(str(scores.mean()) +"\n")
    
    f.write(" error for original data: "+ "\n")
    f.write(str(scores_o) +"\n")
    f.write("mean error for orginal data: "+ "\n")
    f.write(str(scores_o.mean()) +"\n")

    
f.close()
   



