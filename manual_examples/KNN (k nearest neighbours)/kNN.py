import numpy as np

class KNearestNeighbours():
    def __init__(self, k):
        self.k = k
        self.train_set_x = None
        self.train_set_y = None
    
    def fit(self,x, y):
        self.train_set_x = np.array(x)
        self.train_set_y = np.array(y)
    
    def get_neighbours(self, point):
        dist = map(self._euclidian_dist, self.train_set_x, [point]*len(self.train_set_x))
        dist = list(enumerate(dist))
        dist.sort(key=lambda x: x[1])
        neighbours = dist[:self.k]
        return neighbours
    
    def get_labels_neighbours(self,neighbours):
        return [self.train_set_y[neighbour[0]] for neighbour in neighbours]
    
    
    def _euclidian_dist(self, a_point, b_point):
        return np.sqrt(np.sum((a_point - b_point)**2))





