import sys
import random

def findbestsplit(data, trainlabels, col, sample_indexes):
    indexes = []
    colvals ={}
    rows =0
    minus =0
    for i in sample_indexes:
        if(trainlabels.get(i)!=None):
            indexes.append(i)
            colvals[i] = data[i][col]
            rows=rows+1 # initalizing the total number of points
            #print("total=",rows)
            if(trainlabels.get(i)==0):
                minus = minus+1 # counting the total number of zero class data points 
    #print(colvals)
    sorted_indexes = sorted(indexes, key=colvals.__getitem__)
    #print("sortedlist:",sorted_indexes)
    lp=0
    rp=minus
    lsize=1
    rsize=rows-1
    if(trainlabels[sorted_indexes[0]]) ==0:
        lp=lp+1
        rp=rp-1
    
    bestsplit=-1
    bestgini=10000
    
    for i in range(1,len(sorted_indexes),1):
        split = (float(colvals[sorted_indexes[i]])+ float(colvals[sorted_indexes[i-1]]))/2
        gini = (lsize/rows)*(lp/lsize)*(1 - lp/lsize) + (rsize/rows)*(rp/rsize)*(1 - rp/rsize)
        #print(colvals[sorted_indexes[i]]," ", colvals[sorted_indexes[i-1]]) 
        #print(gini," ",split)
        if(gini<bestgini):
            bestgini=gini
            bestsplit=split
        if(trainlabels[sorted_indexes[i]]==0):
            lp=lp+1
            rp=rp-1
        
        lsize=lsize+1
        rsize=rsize-1
    
    return(bestsplit,bestgini)


#---Reading the datafile---#
#datafile = sys.argv[1]
#f=open(datafile,'r')
f = open("datafile.txt")
data =[]
i=0;
l=f.readline()
while(l!=''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(a[j])
    data.append(l2)
    l=f.readline()
    
rows= len(data)
cols=len(data[0])

#trainlabelfile = sys.argv[2]
#f=open(trainlabelfile,'r')
f = open("trainlabelfile.txt")

trainlabels = {}
l=f.readline()

while(l!=''):
    a=l.split()
    trainlabels[int(a[1])]=int(a[0])
    l=f.readline()

    
#Extracting row ids of just training set   
traindata_indexes=[]    
for i in range(0,len(data),1):
    if(trainlabels.get(i)!=None):
        traindata_indexes.append(i)
#print(traindata_indexes)

pred=[]
for count in range(0,100,1):
    #print("iter=",count)
    row_indexes=[random.choice(traindata_indexes) for _ in traindata_indexes]
    #print(row_indexes)
    topsplit=10000
    topcol=-1
    topgini=100000
    for j in range(0,len(data[0]),1):
        bestsplit,bestgini = findbestsplit(data, trainlabels, j, row_indexes)
        #print(bestsplit,bestgini,j)
        if(bestgini<topgini):
            topgini=bestgini
            topcol=j
            topsplit=bestsplit
    
    
            
    #print( "k = ", topcol, "  S = ", topsplit) # for one sample set this compeltes decision stunp



#classifying  if left is zero class or right side , by checking which class's maxiumun datapoints lie of LHS of the topsplit

    zeros_class=0
    ones_class=0
    for i in range(0,len(data),1):
        if(trainlabels.get(i)!=None):
            if(float(data[i][topcol])<topsplit):
                if(trainlabels[i]==0):
                    #print(data[i][topcol])
                    zeros_class = zeros_class+1
                else:
                    ones_class = ones_class+1
                
    #print(zeros_class,ones_class)            
    if(zeros_class > ones_class):
        left=0
        right=1
    else:
        left=1
        right=0
    
    result=[topcol,topsplit,left,right]
    pred.append(result)
    
                

prediction = {}
#print("prediction:")
#print(pred)

for i in range(0,len(data),1):
    if(trainlabels.get(i) == None):
        pred_value = []
        for p in range(0,len(pred),1):
            col = pred[p][0]
            s = pred[p][1]
            if(float(data[i][col]) < s):
                if(pred[p][2]==0):
                    pred_value.append(0)
                if(pred[p][2]==1):
                    pred_value.append(1)
            else:
                if(pred[p][3]==0):
                    pred_value.append(0)
                if(pred[p][3]==1):
                    pred_value.append(1)
                
        #print("for k = ",k," len = ",len(pred_value))
        prediction[i] = pred_value


#print(prediction)

for k in prediction.keys():
    pred_list = prediction[k]
    print("For " ,k , " : " ,max(set(pred_list), key=pred_list.count))

                

