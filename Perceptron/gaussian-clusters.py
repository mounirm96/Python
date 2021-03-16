import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def decision(v):
    return 1 if v >=0  else -1

def predict(x,W):
    decision(W[0])
    return  -(W[0]+W[1]*x)


#def perceptron(cluster1, cluster2, weights, learn_rate):
def perceptron(X, Y):
        weight = np.zeros(len(X[0]))
        learn_rate = 0.1
        nb_train = 1
        for x in range(nb_train):
            for j in range (len(X)):
                if(1*np.dot(weight,X[j]<0)):
                    weight=weight+learn_rate*X[j]
                if(-1*np.dot(weight,Y[j])<=0):
                    weight=weight+learn_rate*Y[j]*-1
        plt.plot([x for x in range(-6,6)],[predict(x,weight) for x in range (-6,6)],label="lineP",linewidth=1)


def test_run(size):
    centre1=np.array([3,3])
    centre2=np.array([-3,-3])
    sigma1=np.array([[1.5,0],[0,1.5]])
    sigma2=np.array([[1.5,0],[0,1.5]])

    taille1=size
    taille2=size
    cluster1=np.random.multivariate_normal(centre1,sigma1,taille1)
    cluster2=np.random.multivariate_normal(centre2,sigma2,taille2)
    



    perceptron(cluster1,cluster2)

    plt.scatter([point[0] for point in cluster1], [point[1] for point in cluster1], color="red")
    plt.scatter([point[0] for point in cluster2], [point[1] for point in cluster2], color="blue")
    plt.scatter(centre1[0], centre1[1], color="green")
    plt.scatter(centre2[0], centre2[1], color="green")


    
    plt.show()
test_run(100)    
