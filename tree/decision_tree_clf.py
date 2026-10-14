import numpy as np
from numpy.typing import NDArray , ArrayLike
from sklearn.metrics import mean_squared_error  , accuracy_score
from sklearn.datasets import make_classification
from collections import Counter

class Node :
    """Creating Nodes for the Decision Tree Regressor"""
    def __init__(self , feature=None , threshold=None , right=None , left=None , is_leaf = False , value = None):
        self.feature = feature # Index of the feature to split on
        self.threshold = threshold # THreshold where the node splits to other nodes
        self.right = right # Point to the Node in the right
        self.left = left # Point to the Node in the left
        self.is_leaf = is_leaf # Is a leaf node (End Node)
        self.value = value # Value of Node - Only if it is a leaf node

class DecisionTreeClassifier:
    def __init__(self , min_samples_split:int=2 , max_depth:int=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self._fitted = False

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        self.X = np.array(X)
        self.y = np.array(y)
        
        self.root_node = self._build_tree(X=self.X , y=self.y , current_depth=0)

        self._fitted = True
    
    def _build_tree(self ,X:ArrayLike=None , y:ArrayLike=None , current_depth:int=0):
        if current_depth >= self.max_depth or X.shape[0] < self.min_samples_split :
            return Node(is_leaf=True,value=y[0]) #Choose the most_common value
       
        best_feature , best_threshold = self._best_split(X , y)
        
        if best_feature == None or best_threshold == None :
            return Node(is_leaf=True ,value=np.mean(y))
        
        left_data = X[: , best_feature] <= best_threshold # Get the boolean values that agree with the threshold
        right_data = X[: , best_feature] > best_threshold

        left = self._build_tree(X[left_data] , y[left_data] , current_depth+1) # Build the left node then the right one
        right = self._build_tree(X[right_data] , y[right_data] , current_depth+1)

        return Node(feature=best_feature ,threshold=best_threshold ,right=right ,left=left)

    def _best_split(self ,X:ArrayLike=None,y:ArrayLike=None):
        best_feature = None
        best_threshold = None
        best_gain = -1 # Do not set to zero to allow setting of the best_feature and threshold
        
        parent_gain = self._gini_impurity(y)
        n_samples, n_features = np.shape(X)

        for feature in range(n_features):
            for threshold in X[: ,feature]:
                # Split dataset based on the threshold value
                left = y[X[: , feature] <= threshold] 
                right = y[X[: , feature] > threshold]
                #left= X[:, feature] <= threshold
                #right= ~left
 
                if len(left) == 0 or len(right)== 0:
                    continue # Skip if there is no useful split

                n_left, n_right = len(left) , len(right)#np.sum(left_mask), np.sum(right_mask)
                child_gini = (n_left / n_samples) * self._gini_impurity(y[left]) + \
                             (n_right / n_samples) * self._gini_impurity(y[right])
                
                # Information gain is reduction in impurity
                gain = parent_gain - child_gini
               
                if gain > best_gain  : # If thew score is greater than the previous one
                    best_gain = gain
                    best_feature , best_threshold = feature , threshold

        return best_feature , best_threshold

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def predict_one(self ,node=None ,X:ArrayLike=None):
        if node.value is not None :
            return node.value

        #print(X[node.feature] <= node.threshold)
        if X[node.feature] <= node.threshold :
            return self.predict_one(node.left , X)

        return self.predict_one(node.right , X)

    def predict(self , X:ArrayLike=None):
        #return np.array([y for y in self.predict_one(self.root_node , np.array(X))])
        return np.array([self.predict_one(self.root_node , x) for x in np.array(X)])

# Get the random values of the datasets
X , y = make_classification()

model = DecisionTreeClassifier()

model.fit(X , y)

y_pred = model.predict(X)

print(y_pred)
#acc = accuracy_score(y , y_pred)

#print(acc)

