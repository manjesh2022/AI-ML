from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width',
'Class'] 

dataset = pd.read_csv("emkmeans.csv",names=names)
dataset.head()

X = dataset.iloc[:,:-1]
label = {'Iris-setosa': 0,'Iris-versicolor': 1, 'Iris-virginica': 2} 
y = [label[c] for c in dataset.iloc[:, -1]]

model=KMeans(n_clusters=3)
model.fit(X)  

print('The accuracy score of K-Mean: ',metrics.accuracy_score(y,
model.labels_) * 100) 

print('The Confusion matrixof K-Mean:\n',metrics.confusion_matrix(y,
model.labels_)) 

gmm=GaussianMixture(n_components=3).fit(X) 
y_cluster_gmm=gmm.predict(X)

print('The accuracy score of EM: ',metrics.accuracy_score(y,
y_cluster_gmm) * 100)

print('The Confusion matrix of EM:\n ',metrics.confusion_matrix(y,
y_cluster_gmm)) 

colormap = np.array(['red','lime','black'])

plt.subplot(1,3,1)
plt.title("KMeans")
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y])
plt.subplot(1,3,2)
plt.title("KMeans")
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[model.labels_])
plt.subplot(1,3,3)
plt.title("EM")
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y_cluster_gmm])
plt.show()

