import numpy as np
from numpy.typing import NDArray , ArrayLike
from sklearn.metrics import mean_squared_error  , accuracy_score
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.model_selection import train_test_split

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
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

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

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_split = None
        best_threshold = float("inf")
        
        # Calculate parent impurity
        parent_gini = self._gini_impurity(y)

        for feat_idx in range(n_features):
            # Get unique sorted values for thresholds
            thresholds = np.unique(X[:, feat_idx])
            
            # If continuous, consider midpoints between unique values
            if len(thresholds) > 1:
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted impurity of children
                n = len(y)
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                child_gini = (n_left / n) * self._gini_impurity(y[left_mask]) + \
                             (n_right / n) * self._gini_impurity(y[right_mask])
                
                # Information gain is reduction in impurity
                gain = parent_gini - child_gini
                if gain > best_gain:
                    best_gain = gain
                    best_split = feat_idx
                    best_threshold = threshold

        return best_split, best_threshold

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

X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size = 0.2)

model = DecisionTreeClassifier()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test , y_pred)

print(acc)
  
