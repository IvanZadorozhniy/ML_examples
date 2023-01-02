import numpy as np
import plotly.express as px
import pandas as pd
x = [[1,1,3],[1,2,4],[2,1,5],[2,2,6],[3,1,7]]
y = [12,25,34,44,56]

X = np.array(x)
Y = np.array(y)

class MultiVariableRegression():
    def __init__(self, theta=np.random.rand(), b=np.random.rand()) -> None:
        self.theta = theta
        self.b = b
        self.x = 0
        self.y = 0
    
    def predict(self, x):
        x = np.array(x)
        return np.dot(x, self.theta) + self.b

    
    def __update_factors(self, learning_rate):
        y_predict = self.predict(self.x)
        coof = ( 1 / len(self.x) )
        error = y_predict - self.y
        
        d_t = coof * self.x.T.dot(error)
        d_b = coof * np.sum(error)
        
        self.theta -= learning_rate * d_t
        self.b -= learning_rate * d_b
        
    def fit(self,x,y,learning_rate=0.01):
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.ones(shape=(len(self.x[0]),))
        self.b = 0
        for i in range(10000):
            self.__update_factors(learning_rate=learning_rate)
            
        


    
   
        



a = MultiVariableRegression()
a.fit(X,Y)
print(a.predict([[1,2,3]]))