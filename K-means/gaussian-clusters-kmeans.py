import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt



#def euc

def generate_center():
    centers=[]
    centre1=np.array([3,3])
    centers.append(centre1)
    centre2=np.array([-3,-3])
    centers.append(centre2)
    centre3=np.array([-6,4])
    centers.append(centre3)
    centre4=np.array([4,-6])
    centers.append(centre4)
    centre5=np.array([-6,-7])
    centers.append(centre5)
    return np.array(centers)



def test_run(size):
    sigma=np.array([[1.5,0],[0,1.5]])
    #centre pour definir les clusters
    centersC=generate_center()

    colors=["red","blue","magenta","yellow","green","black","maroon","turquoise","deeppink","steelblue"]

    cluster1=np.random.multivariate_normal(centersC[0],sigma,size)
    cluster2=np.random.multivariate_normal(centersC[1],sigma,size)
    cluster3=np.random.multivariate_normal(centersC[2],sigma,size)
    cluster4=np.random.multivariate_normal(centersC[3],sigma,size)
    cluster5=np.random.multivariate_normal(centersC[4],sigma,size)

    data=np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5), axis=0)

    

    # Generate random centers
    K = 5
    n = data.shape[0]
    c = data.shape[1]

    n_iter=50

    centers=np.array([]).reshape(c,0) 

    for i in range(K):
        rand=rd.randint(0,n)
        centers=np.c_[centers,data[rand]]

    for i in range(n_iter):
        #calculer la distance euclidienne de chaque point à tous les centers et stocker dans la matrice et
        #trouverer la distance minimale et stocker l'indice de la colonne dans un vecteur C.
        EuclidianDistance=np.array([]).reshape(n,0)
        for k in range(K):
            tempDist=np.sum((data-centers[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
        #regrouper les points de données en fonction de l'index de cluster C
        centers_data={}
        for k in range(K):
            centers_data[k+1]=np.array([]).reshape(2,0)
        for i in range(n):
            centers_data[C[i]]=np.c_[centers_data[C[i]],data[i]]
        
        for k in range(K):
            centers_data[k+1]=centers_data[k+1].T

        for k in range(K):
            centers[:,k]=np.mean(centers_data[k+1],axis=0)
        Output=centers_data



    #plt.scatter(data[:,0], data[:,1], s=7,color=colors[5])
    labels=['cluster 1','cluster 2','cluster 3','cluster 4','cluster 5','cluster 6','cluster 7','cluster 8','cluster 9','cluster 10']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],s=15,c=colors[k],label=labels[k])
    plt.scatter(centers[0,:],centers[1,:],marker='P',s=50,c='lime',label='centers')
    plt.legend()
    plt.show()
    #plt.scatter(centers[:,0], centers[:,1], marker='P', c='#4114CC', s=50)


test_run(100)    
