import numpy as np
from numpy.typing import ArrayLike , NDArray
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class MultinomialNB :
    def __init__(self , alpha:int=1):
        self.alpha = alpha

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        n_samples , n_features = self.X.shape
        
        self._classes = np.unique(self.y)
        self.n_classes = len(self._classes)

        self.priors = np.ones(self.n_classes)

        for idx , c in enumerate(self._classes):
            self.priors[idx] = np.sum(self.y == c) / n_samples
        
        self.feature_count = np.zeros((n_classes , n_features))


        return self

data = pd.read_csv("email.csv")[:-1]
data["Category"] = [1 if i=="spam" else 0 for i in data["Category"]]

vectorizer = CountVectorizer()

X ,y = vectorizer.fit_transform(data["Message"]).toarray() , data["Category"]

X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2 , shuffle=True)

model = MultinomialNB()

model.fit(X_train , y_train)


