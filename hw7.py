# K means clustering is applied to normalized ipl player data

import sys
import math


class K_Means:
    def __init__(self, k =3, tolerance = 0.0001, max_iterations = 5):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data, no_clusters):
        self.k=no_clusters
        self.centroids = {}
        cols = len(data[0])
        #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(0,self.k,1):
            test=[]
            for j in range(0,cols,1):
                test.append(float(data[i][j]))
            self.centroids[i] = test
        #print(self.centroids)

        #begin iterations
        for i in range(self.max_iterations):
            self.classestest={} # for storing rowids belonging to corresponding cluster
            
            for i in range(self.k):
                self.classestest[i] = [] 
            row=0
            #find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                dislist = {}
                for k in range (0,len(self.centroids),1):
                    dis=0
                    for j in range(0,len(data[0]),1):
                        dis =dis+ math.pow((float(data[row][j]) - float(self.centroids[k][j])),2)
                    dislist[k] = math.sqrt(dis) # for kth cluster the datapoint's distance stored 
                index = min(dislist, key=dislist.get) # get min dis's index, that tells datapoint belongs to which cluster
                self.classestest[index].append(row) # adding just row id of data to that particular cluster
                row=row+1

            #average the cluster datapoints to re-calculate the centroids
            vindex=0

            self.centroids_new ={}
            for classification in self.classestest:
                clusterlist = self.classestest.get(vindex)
                listavg_cols=[]    
                for j in range (0,cols,1):
                    sum=0
                    
                    for lrows in clusterlist:
                        sum = sum + float(data[lrows][j])
                    
                    avg=sum/len(clusterlist)
                    listavg_cols.append(avg)

                self.centroids_new[classification]=listavg_cols
                        
                vindex=vindex+1
                

            isOptimal = True
            
            centroid_index=0
            for centroid in self.centroids:
                original_centroid_list= self.centroids[centroid_index]
                new_centroid_list = self.centroids_new[centroid_index]
                if(original_centroid_list!=new_centroid_list):
                    isOptimal = False
                
                centroid_index = centroid_index+1

            #break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            if isOptimal==True:
                print("final cluster:")
                print(self.classestest)
                
                break
            self.centroids = dict(self.centroids_new)




#***************************Main Program*******************************************************************************************    
datafile= sys.argv[1]
f = open(datafile,'r')
#f = open("datafile.txt")

no_clusters = int(sys.argv[2])
#no_clusters=4

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
#print(data)
#print("row=", rows, "cols= ", cols)
#for i in range(0,rows,1):
#    print(data[i])

km = K_Means(3)
km.fit(data,no_clusters)
