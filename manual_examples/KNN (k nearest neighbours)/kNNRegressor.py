import numpy as np
from kNN import KNearestNeighbours


class KNearestNeighboursRegressor(KNearestNeighbours):
    def predict(self,x):
        answers = []
        for sample in x:
            neighbours = self.get_neighbours(sample)
            answer = self.get_labels_neighbours(neighbours)
            print(answer)
            avg = sum(answer)/self.k
            answers.append(avg)
        return np.array(answers)


if __name__ == "__main__":
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
    labels = [1,2,3,1,2,3,4,1]
    k = KNearestNeighboursRegressor(3)
    k.fit(x,labels)
    predict = k.predict([[1.4,1.4]])
    print(predict)