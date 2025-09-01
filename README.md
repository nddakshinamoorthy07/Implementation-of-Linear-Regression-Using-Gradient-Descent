# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DAKSHINA MOORTHY N D
RegisterNumber:  212224230049
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="1086" height="691" alt="image" src="https://github.com/user-attachments/assets/1d3991ae-1bd3-42b6-8bea-17f571d3aecc" />

<img width="893" height="227" alt="image" src="https://github.com/user-attachments/assets/333b902c-357b-47dc-bd47-62f4fb7b7ef2" />

<img width="336" height="877" alt="image" src="https://github.com/user-attachments/assets/fb1b860a-4894-427b-9e0c-f41c0c3d367b" />

<img width="819" height="183" alt="image" src="https://github.com/user-attachments/assets/1ac6bb98-9659-4c4d-a192-7cf78a051fbb" />

<img width="584" height="873" alt="image" src="https://github.com/user-attachments/assets/30d5287c-ad2c-48b1-a800-38497185e785" />

<img width="1018" height="240" alt="image" src="https://github.com/user-attachments/assets/20cee9a1-5dfc-4610-a2d0-bbf5bc7655db" />

<img width="1197" height="274" alt="image" src="https://github.com/user-attachments/assets/66ea2f7e-db39-4d85-be4b-544c08d14a86" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
