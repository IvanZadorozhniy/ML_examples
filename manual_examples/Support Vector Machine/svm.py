
import numpy as np


class SVM:

    def __init__(self, C=1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

    def hingeloss(self, w, b, x, y):
        '''
        Usinge for calculating hinge loss
        link: https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/
        '''
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]

    def fit(self, X, Y, learning_rate=0.001, epochs=1000):
        self.w = np.zeros(len(X[0]))
    
        # Gradient Descent logic
        for epoch in range(1,epochs):
            for i, x in enumerate(X):
                if (Y[i] * np.dot(X[i], self.w)) < 1:
                    self.w = self.w + self.C* ((X[i] * Y[i]) + (-2 * (1/epoch)*self.w))
                else:
                    self.w = self.w +self.C * (-2 * (1/epoch)*self.w)
                


    def predict(self, X):
        print(self.w)

        prediction = np.dot(X, self.w) + self.b  # w.x + b
        return prediction

if __name__ == "__main__":
    x = np.array([
        [1,6],
        [2,4],
        [4,3],
        [6,2],
        [-2,4],
        [4,1],
    ])
    labels = [0,0,0,0,1,1]
    support_vector_machine = SVM()
    support_vector_machine.fit(x,labels)
    predict = support_vector_machine.predict([[2,6]])
    print(predict)