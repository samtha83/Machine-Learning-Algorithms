from array import array
import sys
import math
from sklearn import linear_model
from sklearn import svm
from math import sqrt
import random


# Function to calculate pearson coefficient for feature selection
def pearson_calc(data,trainlabels, val):
    
    r=[]
    
    for j in range(0,len(data[0]),1):
        r.append(0.0)
    #========================calculating mean of each feature column=====================================
    m_X = [] # single rowlist
    m_Y = [] # creating list as each column's mean we need caculate.
    n_X =0; #for counting number of records in class 0
    n_Y = 0; #for counting number of records in class 1
    for j in range(0,len(data[0]),1):
        m_X.append(0)
        m_Y.append(0)
    
    #Mean of columns
    for j in range(0,len(data[0]),1):
        for i in range(0,len(data),1):
            #n_X =n_X+1
            m_X[j]=(m_X[j])+ int(data[i][j])

    #Mean of Output
    for i in range(0,len(data),1):
        n_Y=n_Y+1
        m_Y[0]=(m_Y[0])+(trainlabels[i])
    
    no_rows=len(data)
    #mean
    for j in range(0,len(data[0]),1):
        m_X[j]= m_X[j]/no_rows
    
    #mean
    m_Y=m_Y[0]/n_Y

        

    #print("mean of col",m_X)
    #print("mean of labels", m_Y)

    #for t in range(0,10,1):
        #print(t,"x_mean",m_X[t])

    #============================calculate pearson for each columns============================

    for j in range(0,len(data[0]),1):
        diff_x=0
        diff_y=0
        num=0
        den_x=0
        den_y=0
        for i in range(0,len(data),1):
            diff_x=float(data[i][j])-float(m_X[j])
            diff_y=float(trainlabels[i])-float(m_Y)
            num=num+(diff_x*diff_y)
            den_x= den_x+(diff_x**2)
            den_y=den_y+(diff_y**2)
        deno=sqrt(den_x)*sqrt(den_y)
        if deno == 0:
            deno = 0.000000000001

        r[j]=abs(num/deno)

    sortlist = sorted(range(len(r)), key= r.__getitem__, reverse=True)
    pearson_sel_cols =sortlist[:val]
    pearson_sel_cols_sorted=sorted(pearson_sel_cols)
    return pearson_sel_cols_sorted


#--------------------------------------------Removing cols of less pearson coeffcient from the traindata -------------------------------------------------------------------

def feature_reduction(data_X,pearson_sel_cols_s):
    
    new_sample_train = []
    
    for row in range(0,len(data_X),1):
        rowlist_train = []
        for col in pearson_sel_cols_s:
            rowlist_train.append(data_X[row][col])
        new_sample_train.append(rowlist_train)
           
    return (new_sample_train) 


#----------Accuracy calculation------------------------------------------------------------------------------

def get_accuracy(prediction_dataset,dataset_truelabels):
    correct=0
    for i in range(0,len(prediction_dataset),1):
        if(prediction_sample_subtest[i]==dataset_truelabels[i]):
            correct = correct+1
        
    accuracy =  (correct/len(prediction_dataset))*100
    return accuracy
    


#------------------Loading data--------------------------------------------------

#---Reading the datafile---#
datafile = sys.argv[1]
f=open(datafile,'r')
#f = open("traindata")
data =[]

i=0;
l=f.readline()
v=0
while(l!=''):
    a=l.split()
    l2=array("f",[])
    for j in range(0,len(a),1):
        l2.append(int(a[j]))
        
    data.append(l2)
    
    l=f.readline()
    


#------------------------Loading Labels---------------------------
trueclass = sys.argv[2]
f=open(trueclass,'r')
#f = open("trueclass.txt")
trainlabels={} #Initializing the dictionary
labels=[]
i=0;
l= f.readline()
   
while(l!=''):
    a = l.split()
    trainlabels[int(a[1])]=int(a[0])
    labels.append(int(a[0]))
    l = f.readline()


#-----------------------Actual Logic-----------------------------------------------
best_accuracy=0.0
best_set_cols=[]
final_C=0.0
for count in range(0,10,1):
    
    print("round: ",count)
    print('Splitting Data')
    ratio=0.90
    length_data=len(data)
    train_size=int(length_data*ratio)
    index_train=random.sample(range(length_data),train_size)
    
    train_subset=[]
    test_subset=[]
    labels_train=[]
    labels_test=[]
    
    for ii in range(len(data)):
        if ii in index_train:
            train_subset.append(data[ii])
            labels_train.append(labels[ii])
        else:
            test_subset.append(data[ii])
            labels_test.append(labels[ii])
        
    #del(data,length_data,train_size,ii,trainLabels)
    print('End of Splitting Data')    

    #row_nos_orginal=[x for x in range(0,len(data),1)]
    
    #Calculating pearson and corresponding best columns
    pearson_sel_cols_s =pearson_calc(train_subset, labels_train, 15)
    print("The top pearson columns: ", pearson_sel_cols_s)
    
    
    #Creation of new training data using the same 90% traindata but with less columns
    sample_90train_data = feature_reduction(train_subset, pearson_sel_cols_s)
    
    #Build the data using 10%traindata + bestcols
    sample_10train_data= feature_reduction(test_subset,pearson_sel_cols_s)
    
    val=[1,.01,.001]
    #Build the model using 90%traindata + bestcols
    #regr1 = svm.LinearSVC()
    #regr1.fit(sample_90train_data, labels_train)
    for v in val:
        clf1 = svm.SVC(kernel='linear', C=v).fit(sample_90train_data, labels_train)
        
        #prediction_sample_subtest = regr1.predict(sample_10train_data)
        prediction_sample_subtest = clf1.predict(sample_10train_data)
        
        #checking accuracy of model
        accuracy_subtest =get_accuracy(prediction_sample_subtest,labels_test)
        print("accuracy of sample test data with C= ",v," = ",accuracy_subtest)
    
        if(accuracy_subtest>best_accuracy):
            best_accuracy=accuracy_subtest
            best_set_cols=pearson_sel_cols_s
            final_C = v
   
    
#----------------------------------------end of cross validation------------------------------------------------------------------------------------   


print("best_set_cols=",best_set_cols)
print("best C=", final_C)
print("best_accuracy=",best_accuracy)

#Creation of new training data using the same 100% traindata but with less columns
sample_train_data= feature_reduction(data,best_set_cols)
    
    
#Build the model using 100%traindata + bestcols
#regr = svm.LinearSVC()
#regr.fit(sample_train_data, labels)   
clf = svm.SVC(kernel='linear', C=final_C).fit(sample_train_data, labels)
     
       
#-----------------------------------------------Loading actual testdata----------------------------------------------------------------------
testdata = sys.argv[3]
f=open(testdata,'r')
#f = open("testdata")
test_data =[]


l=f.readline()
 
while(l!=''):
    a=l.split()
    l2=array("f",[])
    for j in range(0,len(a),1):
        l2.append(int(a[j]))
        
    test_data.append(l2)
    
    l=f.readline()
    
rows_test= len(test_data)
cols_test=len(test_data[0])

#------------------------------------------------------updating testdata by applying best cols--------------------------

test_data_mod = []

for row in range(0,rows_test,1):
    rowlist_test = []
    for col in best_set_cols:
        rowlist_test.append(test_data[row][col])
    test_data_mod.append(rowlist_test)

        
#print(len(test_data_mod), " ", len(test_data_mod[0]))


#---------------------- Running the prediction for test by using Model(buitl by 100% traindata and best cols)
#final_prediction = regr.predict(test_data_mod)
final_prediction = clf.predict(test_data_mod)

#----------------------------------------Printing the Results--------------------
f = open("test_data.labels","w")
for row_no in range(0,len(final_prediction),1):
    val=str(final_prediction[row_no])
    ##print(val)
    row=str(row_no)
    f.write(val+"  "+row+ "\n")

f.close() 

#-----------------Printing the Results on console---------------------------------
print("Predictions:")
for row_no in range(0,len(final_prediction),1):
    val=str(final_prediction[row_no])
    ###print(val)
    row=str(row_no)
    print(val+"  "+row)