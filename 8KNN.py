import pandas as pd

dataset=pd.read_csv("iris.csv")
col=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

x=dataset[col].values
y=dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
   
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print("y_pred y_test")
for i in range(len(y_pred)):
    print(y_pred[i]," ", y_test[i])

from sklearn.metrics import confusion_matrix
cmm=confusion_matrix(y_pred,y_test)
print("Confusion Matrix i:",cmm)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,y_test)*100
print("Accuracy Score of Model is",str(round(acc,2),"%"))
