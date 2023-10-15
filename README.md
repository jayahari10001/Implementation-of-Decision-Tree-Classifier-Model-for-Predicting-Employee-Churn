Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import the required libraries.
Upload and read the dataset.
Check for any null values using the isnull() function.
From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
Find the accuracy of the model and predict the required values by importing the required module from sklearn.
Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Nithish Kumar P
RegisterNumber: 212221040115


import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy value:")
accuracy

print("data prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
Output:
Screenshot (63)

Screenshot (64)

Screenshot (65)

Screenshot (66)

Screenshot (67)

Screenshot (68)

Screenshot (69)

Screenshot (70)

Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
