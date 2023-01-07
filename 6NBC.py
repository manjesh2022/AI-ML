import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("NBC.csv")
print("The first five rows of the data are ",data.head())

X = data.iloc[:,:-1]
print("The first five rows of the train data are ",X.head())

y = data.iloc[:,-1]
print("The output of train data is ",y.head())

X_Pregnancies = X.iloc[:,0]
X_Glucose = X.iloc[:,1]
X_BloodPressure = X.iloc[:,2]
X_SkinThickness = X.iloc[:,3]
X_Insulin = X.iloc[:,4]
X_BMI = X.iloc[:,5]
X_DiabeticPedigreeFunction = X.iloc[:,6]
X_Age = X.iloc[:,7]

le_Pregnancies = LabelEncoder()
X_Pregnancies = le_Pregnancies.fit_transform(X_Pregnancies)
le_Glucose = LabelEncoder() 
X_Glucose = le_Glucose.fit_transform(X_Glucose) 
le_BloodPressure = LabelEncoder() 
X_BloodPressure = le_BloodPressure.fit_transform(X_BloodPressure) 
le_SkinThickness = LabelEncoder() 
X_SkinThickness = le_SkinThickness.fit_transform(X_SkinThickness) 
le_Insulin = LabelEncoder() 
X_Insulin = le_Insulin.fit_transform(X_Insulin) 
le_BMI = LabelEncoder() 
X_BMI = le_BMI.fit_transform(X_BMI) 
le_DiabeticPedigreeFunction = LabelEncoder() 
X_DiabeticPedigreeFunction = le_DiabeticPedigreeFunction.fit_transform(X_DiabeticPedigreeFunction) 
le_Age = LabelEncoder() 
X_Age = le_Age.fit_transform(X_Age)

print("\nNow the output of the train data is \n",X.head())

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB().fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print("The accuracy score is ",accuracy_score(classifier.predict(X_test), y_test)*100)
