import numpy as np
import plotly.express as px
import pandas as pd
x = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]]
y = [[10,14],[20,23],[30,33],[40,42],[50,60]]

X = np.array(x)
Y = np.array(y)

# TODO: Add comments and math description
class MultiVariableRegression():
    def __init__(self, theta=np.random.rand(), b=np.random.rand()) -> None:
        self.theta = theta
        self.b = b
        self.x = 0
        self.y = 0
    
    def predict(self, x):
        x = np.array(x)
        return np.dot(x, self.theta) + self.b
  
        
    def fit(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = 0
        self.theta = np.linalg.matrix_power((self.x.T.dot(self.x)),-1).dot(self.x.T).dot(self.y)

        
        


    
   
        



a = MultiVariableRegression()
a.fit(X,Y)
print(a.predict([[1,2,5]]))