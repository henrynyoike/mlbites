import numpy as np
from numpy.typing import NDArray , ArrayLike

class LinearRegression: 
    def __init__(self , 
            fit_intercept:bool=True):
        
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept_ = fit_intercept

    def fit(self , X:ArrayLike=None , 
            y:ArrayLike|DataFrame=None ,
            learning_rate:float = 0.001,
            iterations:int=100):
        """Fit the Linear Regression Model"""
        self.X = np.asarray(X , dtype=float)
        self.y = np.asarray(y , dtype=float)
        
        # Get the shapes of X
        n_samples , n_features = X.shape
            
        if n_samples != self.y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples")
   
        X_mean = np.mean(self.X , axis=0)
        y_mean = np.mean(self.y)
        
        if self.fit_intercept_ :
            self.X = self.X - X_mean
            self.y = self.y - y_mean
            
        # Get the coefficient using the Normal Equation or the Least Squares Function
        #self.coef_ = np.linalg.solve(self.X.T@self.X + (self.alpha * np.eye(n_features)) , (self.X.T@self.y))
        self.coef_ , *_ = np.linalg.lstsq(self.X , self.y , rcond=None)
        
        if self.fit_intercept_ :
            self.intercept_ = y_mean - (X_mean @ self.coef_)
        else :
            self.intercept_ = 0

        return self.coef_ , self.intercept_
    
    def predict(self , X:ArrayLike):
        X = np.asarray(X , dtype=float)
        return np.array(np.dot(X , self.coef_) + self.intercept_)

