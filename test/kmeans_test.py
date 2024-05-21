from sklearn.cluster import KMeans
import numpy as np

x = np.array([1,0,10,3,7,8,7,6,9])
x = x.reshape(-1,1)
kmeans = KMeans(n_clusters=2,random_state=0).fit(x)
print (kmeans.labels_)