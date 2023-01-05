
import numpy as np


class SVM:

    def __init__(self, C=1.0):
        # C = error term
        self.C = C
        self.w = None
        self.b = None

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

    def fit(self, X, Y, learning_rate=0.001, epochs=10000):
        X = np.array(X)
        Y = np.array(Y)
        y = np.where(Y <= 0, -1, 1)
        self.w = np.random.rand(len(X[0]))
        self.b = 0

        # Gradient Descent logic
        for epoch in range(1, epochs):
            for idx, x in enumerate(X):
                cond = Y[idx] * (np.dot(x, self.w) - self.b)
                if cond >= 1:
                    self.w += learning_rate * 2 * self.C * self.w
                    self.b += learning_rate * y[idx]
                else:
                    self.w -= learning_rate * \
                        (2 * self.C * self.w - np.dot(x, y[idx]))
                    self.b -= learning_rate * y[idx]

    def predict(self, X):
        X = np.array(X)
        print(self.w)
        print(self.b)

        prediction = np.sign(np.dot(X, self.w) + self.b)  # w.x - b
        return np.where(prediction == -1, 0, 1)


if __name__ == "__main__":
    from sklearn.svm import LinearSVC
    
    x = np.array([
        [1, 6,2],
        [2, 4,3],
        [3, 3,4],
        [-2, 2,5],
        [-2, 4,2],
        [-4, 1,3],
    ])
    test = [[-1,2,2],[2,3,1],[1,2,3],[4,4,-1],[-4,4,3],[1,2,3]]
    labels = [0, 0, 0, 1, 1, 1]
    support_vector_machine = SVM()
    support_vector_machine.fit(x, labels)
    predict = support_vector_machine.predict(test)
    print(predict)
    clf = LinearSVC()
    clf.fit(x, labels)
    predict = clf.predict(test)
    print(predict)
    print(clf.get_params())
    print(clf.coef_)
    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame({"X":x[:,0],"Y":x[:,1],"Z":x[:,2], "labels":labels})
    fig = px.scatter_3d(df, x='X', y='Y', z='Z',
                color='labels')
    fig.add_shape(type="line", x0=0,y0=30,x1=10,y1=10, line={"color":"Red"})
    fig.show()