import numpy as np
from numpy.typing import ArrayLike
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class DBSCAN :
    def __init__(self , eps:int|float=0.5 , min_samples:int =5):
        self.eps = eps
        self._fitted = False
        self.clusters = None
    
    def euclidean_distance(self , point_a , point_b):
        """Calculate the Euclidean Distance between 2 points"""
        return np.sqrt(np.sum(point_a - point_b) ** 2)

    def get_closest_points(self , X:ArrayLike=None , point:ArrayLike=None):
        """A recursive function that gets all the points closest by density to a point and stores them"""
        ungrouped_X = []
        
        # Loop through every cluster
        for cluster_idx , cluster in enumerate(self.clusters):
            for x in X :
                if len(cluster) > 0 :
                    for value in cluster :
                        if value.all() == x.all() :
                            continue
                distance = self.euclidean_distance(x , point)
                if distance <= self.eps :
                    self.clusters[cluster_idx].append(x)
                        # If alreaady in a group set it as a core point and get the closest points
                    if len(ungrouped_X) != 0 :
                            self.get_closest_points(ungrouped_X , x)
                else:
                    ungrouped_X.append(x)
        
        samples = len(ungrouped_X)
        
        if len(ungrouped_X) > 0 :
            random_point = ungrouped_X[np.random.choice(range(samples))]
        
            self.get_closest_points(ungrouped_X , random_point)

    def fit(self , X:ArrayLike=None ):
        """Fit the DBSCAN Model and define clusters"""
        self.X = np.asarray(X)

        self.n_samples , self.n_features = self.X.shape
        
        # Create a large set of clusters to use then clean later
        self.clusters = [[] for i in range(10)]

        random_point = (self.X[np.random.choice(self.n_samples)])
        
        self.get_closest_points(self.X , random_point)

        self._fitted = True


X , y = make_blobs(n_features=2 , n_samples=100)

model = DBSCAN()

model.fit(X)
#plt.scatter(X ,y)
#plt.show()
