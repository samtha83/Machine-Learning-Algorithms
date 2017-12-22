import sys
from math import sqrt
import random

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
    data[i].append(1)
    i=i+1;
    l = f.readline()
    
rows = len(data)
cols = len(data[0])

#print("row=", rows, "cols= ", cols)
#for i in range(0,rows,1):
#    print(data[i])


#==========================#

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
        trainlabels[int(a[1])]=int(a[0])-1
    else:
        trainlabels[int(a[1])]=int(a[0])
    l = f.readline()

#==========================#

#Initializing w
w=[]

for j in range(0, cols, 1):
    w.append(0)

for j in range(0, cols, 1):
    w[j]=w[j] + .02*random.random()- .01 # random.random() -Random float x, 0.0 <= x < 1.0

#for j in range(0, cols, 1):
 #   print(w[j])
    
#---Gradient Descent iteration-------

#---Subroutine to calculate dot product---

def dot_product(w,data):
    sum_dp=0
    for j in range(0,cols,1):
        sum_dp = sum_dp+ w[j]*float(data[j])
    return sum_dp


eta = .001 #learning rate

for l in range(0,10000,1):
    #Compute differential of f
    df = []
    for j in range(0, cols, 1):
        df.append(0)
    
    for i in range(0,rows,1):
        dp=0
        if(trainlabels.get(i)!=None):
            dp=dot_product(w,data[i])
            for j in range(0,cols,1):
                df[j] = df[j] + (trainlabels[i] -dp)*float(data [i][j])
    
    #Update w
    for j in range(0,cols,1):
        w[j]=(w[j]) + eta*(df[j])

    #calculate error
    error =0.0
    for i in range(0,rows,1):
        if(trainlabels.get(i)!=None):
            error = error+ float((trainlabels[i]-dot_product(w,data[i]))**2)
  
    
print("error",error) #Final error after convergence

print("w=")
          
normw=0
for j in range(0,cols-1,1):
    normw=normw+w[j]**2
    print(w[j])
    
normw=sqrt(normw)
print("||w||",normw)
d_orgin = abs(w[len(w)-1]/normw)
print("absolute distance to origin",d_orgin)

#------Prediction-----------

for i in range(0,rows,1):
    if(trainlabels.get(i)==None):
        dp=dot_product(w,data[i])
        if(dp>0):
            print("1",i)
        else:
            print("0",i)