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
        trainlabels[int(a[1])]=int(a[0])-1 #storing 0 the label as -1
    else:
        trainlabels[int(a[1])]=int(a[0])
    l = f.readline()

#==========================#

#Initializing w
w=[]

for j in range(0, cols, 1):
    w.append(0)	

for j in range(0, cols, 1):
    w[j]=w[j] + .02*random.uniform(0,1)- .01 # random.random() -Random float x, 0.0 <= x < 1.0

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

prevobj=1000000
obj = 10
while(abs(prevobj-obj)>.000000001):
    #Compute differential of f
    df = []
    for j in range(0, cols, 1):
        df.append(0)
    
    for i in range(0,rows,1):
        dp=0
        if(trainlabels.get(i)!=None):
            dp=dot_product(w,data[i])
            if(dp*trainlabels[i]>=1):
                df[j] =df[j]+0; #differention is zero
            else:
                for j in range(0,cols,1):
                    df[j] = df[j] + (trainlabels[i])*float(data [i][j])
    
    #Update w
    for j in range(0,cols,1):
        w[j]=(w[j]) + eta*(df[j])

    #calculate error
    error =0.0
    for i in range(0,rows,1):
        if(trainlabels.get(i)!=None):
            check = 1-float(trainlabels[i]*dot_product(w,data[i])) # hinge loss calculation
            if(check<0):
                error = error + 0;
            else:
                error = error + check;
                
    prevobj = obj
    obj = error
            
  
    
#print("error",error) #Final error after convergence

print("w=")

for j in range(0,cols-1,1):
    print(w[j])

print("w0= ", w[cols-1])
    
          
normw=0
for j in range(0,cols-1,1): ## removed cols-1 as I need all w1,w2 and W0...remember w0 is last columnS,as ||w|| =sqrt(w1^2 +w2^2)
    normw=normw+w[j]**2
    #print(w[j])
    
normw=sqrt(normw)
#print("||w||",normw)
d_orgin = abs(w[len(w)-1]/normw)
print("distance to origin",d_orgin)

#------Prediction-----------

for i in range(0,rows,1):
    if(trainlabels.get(i)==None):
        dp=dot_product(w,data[i])
        if(dp>0):
            print("1",i)
        else:
            print("0",i)
