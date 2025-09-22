# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Data preprocessing:
3. Cleanse data,handle missing values,encode categorical variables.
4. Model Training:Fit logistic regression model on preprocessed data.
5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6. Prediction: Predict placement status for new student data using trained model.
7. End the program.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: AKSHAY KARTHICK ASR
RegisterNumber: 212224230015
```

```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Placement_Data.csv")
dataset = dataset.drop(['sl_no', 'salary'], axis=1)

categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'degree_t', 'workex', 'specialisation', 'status', 'hsc_s']
for col in categorical_cols:
    dataset[col] = dataset[col].astype('category').cat.codes

print("Dataset:")
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

theta = np.random.randn(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    return np.where(h >= 0.5, 1, 0)

y_pred = predict(theta, X)
accuracy = np.mean(y_pred == y)
print("\nAccuracy:", accuracy)
print("\nY_pred:\n", y_pred)
print("\nY:\n", y)

xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew1_scaled = scaler.transform(xnew1)
print("\nY_prednew:", predict(theta, xnew1_scaled))

xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
xnew2_scaled = scaler.transform(xnew2)
print("Y_prednew:", predict(theta, xnew2_scaled))

```

## Output:
<img width="710" height="745" alt="image" src="https://github.com/user-attachments/assets/f050d381-b088-4de3-98bf-d93ae8a0d594" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

