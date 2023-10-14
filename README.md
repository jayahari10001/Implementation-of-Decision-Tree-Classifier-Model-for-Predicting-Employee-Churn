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
Developed by: JAYAHARI E
RegisterNumber: 212221040065


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
![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/3137319c-a85a-4f53-a4db-bfc2abafab22)

![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/7d9ff341-8df9-4b1c-81e4-093f6446c009)


![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/04647e16-56b0-4eb2-a0be-2ad51a69a782)


![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/cf524bd0-b069-405c-8185-417286e0305a)

![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/1bbce767-9d63-4f7d-a166-b08af0c810db)

![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/2a580a1e-b8f4-4151-b1ea-31fd54de91bd)


![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/a27c246f-0355-4a07-97d7-89ebc292da25)


![image](https://github.com/jayahari10001/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115681467/0beeee94-bfdc-4035-ae4a-958fd0374c43)


Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
