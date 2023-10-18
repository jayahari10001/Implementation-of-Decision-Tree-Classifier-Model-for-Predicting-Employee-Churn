# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```

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
```

## Output:
![Screenshot (63)](https://github.com/Jayahari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/01bfa1f4-37c8-4535-9051-44a682da33a8)

![Screenshot (64)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/876384bc-f6f9-4f4f-bd3c-f4d9afbb2d8b)

![Screenshot (65)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/3f9b9eb7-9cf2-4d50-a8c8-8d1b27a51ac2)

![Screenshot (66)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/c60e5937-5798-4bdd-b8df-23811ed571b9)

![Screenshot (67)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/723ca951-984c-4f9e-96fe-eb129583ed4c)

![Screenshot (68)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/6cd84e93-41cb-4b07-a490-916ca14ef8ec)

![Screenshot (69)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/35d459a9-a423-463c-b5f3-bb2f682b6710)

![Screenshot (70)](https://github.com/Naadira/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135126/5f1ecd03-3061-492e-8631-95f25f83f1b0)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

