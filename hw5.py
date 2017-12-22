import sys



def findbestsplit(data, trainlabels, col):
    indexes = []
    colvals ={}
    rows =0
    minus =0
    for i in range(0,len(data),1):
        if(trainlabels.get(i)!=None):
            indexes.append(i)
            colvals[i] = data[i][col]
            rows=rows+1 # initalizing the total number of points
            if(trainlabels.get(i)==0):
                minus = minus+1 # counting the total number of zero class data points 
    #print(colvals)
    sorted_indexes = sorted(indexes, key=colvals.__getitem__)
    
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
datafile = sys.argv[1]
f=open(datafile,'r')
#f = open("datafile_o.txt")
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

trainlabelfile = sys.argv[2]
f=open(trainlabelfile,'r')
#f = open("trainlabelfile_o.txt")

trainlabels = {}
l=f.readline()

while(l!=''):
    a=l.split()
    trainlabels[int(a[1])]=int(a[0])
    l=f.readline()

topsplit=10000
topcol=-1
topgini=100000
for j in range(0,len(data[0]),1):
    bestsplit,bestgini = findbestsplit(data, trainlabels, j)
    if(bestgini<topgini):
        topgini=bestgini
        topcol=j
        topsplit=bestsplit
        
#classifying  if left is zero class or right side , by checking which class's maxiumun datapoints lie of LHS of the topsplit

print( "k = ", topcol, "  S = ", topsplit)