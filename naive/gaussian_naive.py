import numpy as np
from numpy.typing import ArrayLike
from sklearn.datasets import make_classification , load_iris
from sklearn.metrics import accuracy_score

class GaussianNB :
    def __init__(self ):
        pass

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        """Fit the Gaussian Naive Bayes Model"""
        self.X = np.array(X)
        self.y = np.array(y)

        self.n_samples , self.n_features = X.shape
        
        self.classes = np.unique(self.y) # Store the original class names
        self.n_classes = len(self.classes)

        self.priors = np.zeros((self.n_classes))
        self.mean = np.zeros((self.n_classes , self.n_features))
        self.variance = np.zeros((self.n_classes , self.n_features))
        
        for idx , class_n in enumerate(self.classes):
            x_class = self.X[y == class_n]
            self.priors[idx] = x_class.shape[0] / self.X.shape[0]
            self.mean[idx , :] = np.mean(x_class , axis=0)
            self.variance[idx , :] = np.var(x_class , axis=0)

        
    def _gaussian_log_pdf(self , x:ArrayLike=None , mean:ArrayLike=int|float , variance:int|float=None):
        """Get the probability density of the values x"""
        #return -(1 / np.sqrt(2 * np.pi * variance) * np.log((x - mean)**2/(2 * variance)))
        return -0.5 * np.log(2 * np.pi * variance) - ((x - mean) ** 2) / (2 * variance)
    def _predict_one(self , X:ArrayLike=None):
        """Predict the probabilities of every class"""
        posteriors = []

        for idx , class_n in enumerate(self.classes):
            self.log_prior = np.log(self.priors[idx])

            self.log_likelihoods = np.sum(
                                    self._gaussian_log_pdf(x=X , mean=self.mean[idx] , variance=self.variance[idx]))
        
            posteriors.append(self.log_prior + self.log_likelihoods)

        return self.classes[np.argmax(posteriors)]

    def predict(self , X:ArrayLike=None , y:ArrayLike=None):
        """Predict the Class of X using the Bayes Theory and the Probability Density Formula"""
        x_pred = np.array(X)
        
        return np.array([self._predict_one(x) for x in x_pred])

data  = load_iris()

X , y = make_classification()#data.data , data.target

model = GaussianNB()

model.fit(X ,y)

y_pred = model.predict(X)

print(accuracy_score(y , y_pred))


