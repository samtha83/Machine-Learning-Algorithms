import sys
from math import sqrt
from math import exp
from math import log
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
    data[i].append(1) # W0 initialization by adding 1 to each row basically creating column of 1's 
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
    trainlabels[int(a[1])]=int(a[0])
    l = f.readline()

#==========================#

#Initializing w
w=[]

for j in range(0, cols, 1):
    w.append(0)	

for j in range(0, cols, 1):
    w[j]=w[j] + random.uniform(-0.01,.01)# random.random() -Random float x, 0.0 <= x < 1.0
    
#w[0]=0.003983141098430903
#w[1]=-0.008135716
#w[2]=-0.002254097

#for j in range(0, cols, 1):
    #print(w[j])
    
#---Gradient Descent iteration-------

#---Subroutine to calculate dot product---

def dot_product(w,data):
    sum_dp=0
    for j in range(0,cols,1):
        sum_dp = sum_dp+ w[j]*float(data[j])
    #print("dp= ",sum_dp)    
    return sum_dp


eta = .01 #learning rate

prevobj=1000000
obj = prevobj-10

while((prevobj-obj)>.0000001):
    #prevobj = obj
    #Compute differential of f
    df = []
    
    for j in range(0, cols, 1):
        df.append(0)
        
    for i in range(0,rows,1):
        dp=0
        d=0
        dp_neg=0
        e=0        
        if(trainlabels.get(i)!=None):
            dp=dot_product(w,data[i])
            dp_neg = -1*dp
            e = exp(dp_neg)
            d= 1+e
            for j in range(0,cols,1): #col=col-1  check as last col is w0 and it has a diff formula
                df[j] = df[j] + ( (trainlabels[i]- (1/d))*float(data [i][j]))
                
         
    #Update w
    for j in range(0,cols,1):
        w[j]=(w[j]) + eta*(df[j])
    


    #calculate error
    error =0.0
    for i in range(0,rows,1):
        dp=0
        d=0
        dp_neg=0
        e=0
        step1=0
        step2=0
        step3=0
        step4=0
        step5=0
        err=0
        step1_0=0
        step2_0=0
        if(trainlabels.get(i)!=None):
            dp=dot_product(w,data[i])
            dp_neg = -1*dp
            e = exp(dp_neg)
            
            d=1+e
            step1_0 = float(1/d)
            #step1= log(step1_0,10)
            step2_0 = float(e/d)
            #step2 = log(step2_0)
            step3=trainlabels[i]*log(step1_0,10)
            step4 = (1-trainlabels[i])*log(step2_0,10)
            step5 = step3+step4
            err =-1*step5
           
        error = error + err
     
  
    prevobj = obj
    obj = error
    #print("prevobj ",prevobj,"  obj: ",obj)
 
print("error",error) #Final error after convergence

print("w=")

for j in range(0,cols-1,1):
    print(w[j])
    
          
normw=0
for j in range(0,cols-1,1): ## removed cols-1 as I need all w1,w2 and W0...remember w0 is last columnS,as ||w|| =sqrt(w1^2 +w2^2)
    normw=normw+w[j]**2
    #print(w[j])
    
normw=sqrt(normw)
print("||w||",normw)
d_orgin = (w[len(w)-1]/normw)
print("distance to origin",d_orgin)

#------Prediction-----------

for i in range(0,rows,1):
    if(trainlabels.get(i)==None):
        dp=dot_product(w,data[i])
        if(dp>0):
            print("1",i)
        else:
            print("0",i)
