import numpy as np
from kNN import KNearestNeighbours



class KNearestNeighboursClassifier(KNearestNeighbours):
    def predict(self,x):
        answers = []
        for sample in x:
            neighbours = self.get_neighbours(sample)
            answer = self.get_labels_neighbours(neighbours)
            answers.append(np.bincount(answer).argmax())
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
    labels = [0,0,0,0,1,1,1,1]
    k = KNearestNeighboursClassifier(2)
    k.fit(x,labels)
    predict = k.predict([[2.5,2.5]])
    print(predict)