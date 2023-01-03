import numpy as np

class NaiveBayesClassifier():
    
    def fit(self, X, Y):
        self.total_rows = len(Y)
        self.statistics_for_class = self.__get_statistics_for_class(X,Y)

    def __get_statistics(self,X):
        def create_stat(col):
            return {
                "mean":np.mean(col), 
                "std":np.std(col), 
                "lenght":len(col)
            } 
        return [create_stat(col) for col in zip(*X)]
    
    def __calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x-mean)**2 / (2 * std**2 )))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def __get_statistics_for_class(self, X,Y):
        stat = {}
        for i, cls in enumerate(Y):
            if cls in stat:
                stat[cls] += [X[i]]
            else:
                stat[cls] = [X[i]]
                
        for cls,data in stat.items():
            stat[cls] = self.__get_statistics(data)
        return stat
    
    def predict_row(self,x):
        probability = {}
        for key, value in self.statistics_for_class.items():
            probability[key] = value[0]["lenght"] / self.total_rows
            for i, column in enumerate(value):
                probability[key] *= self.__calculate_probability(x[i], column["mean"], column["std"])
        return probability
    
    def predict(self,X):
        probabilities = [self.predict_row(row) for row in X]

        return [
            max(prob.items(), key=lambda x: x[1])[0] 
            for prob in probabilities
        ]
            

x = np.array([
        [1,1],
        [0,0],
        [1,0],
        [0,1],
        [2,1],
        [2,2],
        [1,2],
        [2,3],
    ])
y = [0,0,0,0,0,1,1,1]

# x = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
# y = [1, 1, 1, 2, 2, 2]

X = np.array(x)
Y = np.array(y)

nb = NaiveBayesClassifier()
nb.fit(X,Y)
print(nb.predict([[2,2],[3,1],[1,2]]))



from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[2,2],[3,1],[1,2]]))

