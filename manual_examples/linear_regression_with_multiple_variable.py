import numpy as np
import plotly.express as px
import pandas as pd
x = [[1],[2],[3],[4],[5]]
y = [12,25,34,44,56]

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
    
    
    def __get_cost_function_value(self):
        y_predict = self.predict(self.x)
        error = np.sum((y_predict - self.y)**2)
        cost_function = ( 1 / len(self.x) ) * 0.5 * error
        return cost_function

    
    def __update_factors(self, learning_rate):
        y_predict = self.predict(self.x)
        coof = ( 1 / len(self.x) )
        error = y_predict - self.y
        
        d_t = coof * self.x.T.dot(error)
        d_b = coof * np.sum(error)
        
        self.theta -= learning_rate * d_t
        self.b -= learning_rate * d_b
    

    def __print_info(self, step):
        print(f"Step #{step}")
        print(f"Cost function = {self.__get_cost_function_value()}")
        print(f"Theta = {self.theta}")
        print(f"b = {self.b}")
  
        
    def fit(self,x,y,steps=1000, learning_rate=0.01, verbose=True):
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.ones(shape=(len(self.x[0]),))
        self.b = 0
        
        for step in range(steps):
            self.__update_factors(learning_rate=learning_rate)
            if verbose and step % 100 == 0:
                self.__print_info(step)
            
        


    
   
        



a = MultiVariableRegression()
a.fit(X,Y,steps=10000)
print(a.predict([[10]]))