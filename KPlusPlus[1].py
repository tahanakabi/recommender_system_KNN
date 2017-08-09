import numpy as np
import random
import pickle
from sklearn.cluster import KMeans
import pandas as pd

class DetK(KMeans):

    def num_clusters(self,X, max):
        self.X=X
        fs, sK = self.fK(2)
        k = 2
        for i in range(3, max):
            f=self.fK(i, Skm1=sK)[0]
            if f < fs:
                fs, sK = self.fK(i, Skm1=sK)
                k = i
        return k

    def find_clusters(self,inputs):
        table = pd.DataFrame(columns=['points', 'labels'],index=range(len(inputs)))
        for idx in range(len(inputs)):
            table.iloc[idx, 0] = inputs[idx]
            table.iloc[idx, 1] = self.labels_[idx]
        clusters=[]
        for i in self.labels_:
            cl=table['points'][table['labels'] == i]
            clusters.append(cl)
        return clusters


    def fK(self, thisk, Skm1=0):
        X = self.X
        self.n_clusters=thisk
        self.fit(X)
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3 / (4 * Nd) if k == 2 else a(k - 1, Nd) + (1 - a(k - 1, Nd)) / 6
        mu = self.cluster_centers_
        clusters=self.find_clusters(X)
        Sk = sum([np.linalg.norm(mu[i] - c) ** 2 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk / (a(thisk, Nd) * Skm1)
        return fs, Sk

pickle_in=open('inputs','r')
inputs=pickle.load(pickle_in)
clf=DetK()
num=clf.num_clusters(inputs, 20)
print num