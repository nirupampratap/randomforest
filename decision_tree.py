from util import entropy, information_gain, partition_classes
import numpy as np 
import ast
import csv
from scipy import stats

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.node_list = {'X':[], 'y':[]}
        self.tree = {'left': None, 'right':None}
        self.node_attrs = {'info_gain': 0, 'split_attr': None, 'split_val': None, 'is_categorical': None, 'is_leaf': False}
        self.class_pred = None
        self.depth = 1

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        
        # Step - 1 Check termination conditions
        # If all Y are the same then return. There is nothing to split

        if(len(set(y)) <= 1):
            self.class_pred = y[0]
            self.node_attrs['is_leaf'] = True
            return
        else:
            x0 = X[0]
            is_same = True
            # If all X are same then take the y value that repeats the most and use it for the class
            for xi in X:
                if(str(xi) != str(x0)):
                    is_same = False
                    break

            if is_same == True and self.depth >= 25:
                self.node_attrs['is_leaf'] = True
                count_y = {0:0.0,1:0.0}
                for i in y:
                    count_y[i] += 1

                if count_y[0] >= count_y[1]:
                    self.class_pred = 0
                else:
                    self.class_pred = 1
                return

        # Step - 2 Loop on all attributes and check which variable can be use for the split
        for j in range(len(X[0])):

            is_categorical = True

            if(len(X) > 0 and type(X[0][j]) is int):
                is_categorical = False

            xj_list = []
            for row in X:
                xj_list.append(row[j])

            xj_unq = sorted(list(set(xj_list)))

            # For non categorical variables let us use the max. Tested with min and mean and then fixed on max and use the first variable for categorical values. Tested with mode and finalized on the first value itself
            if(is_categorical == False):
                xvar = np.max(xj_unq)
            else:
                xvar = xj_unq[0]
                #xvar = stats.mode(xj_unq)
            
            x_left, x_right, y_left, y_right = partition_classes(X, y, j, xvar)
              
            new_info_gain = information_gain(y, [y_left,y_right])

            if(new_info_gain > self.node_attrs['info_gain']):
                self.node_attrs['info_gain'] = new_info_gain
                self.node_attrs['split_attr'] = j
                self.node_attrs['split_val'] = xvar
                self.node_attrs['is_categorical'] = is_categorical

                left_node = DecisionTree()
                left_node.node_list['X'] = x_left
                left_node.node_list['y'] = y_left
                left_node.depth = self.depth + 1

                right_node = DecisionTree()
                right_node.node_list['X'] = x_right
                right_node.node_list['y'] = y_right
                right_node.depth = self.depth + 1

                self.tree['left'] = left_node
                self.tree['right'] = right_node

        # Step - Call the same function on left child and right child
        if self.tree['left'] is not None:
            self.tree['left'].learn(self.tree['left'].node_list['X'], self.tree['left'].node_list['y'])

        if self.tree['right'] is not None:
            self.tree['right'].learn(self.tree['right'].node_list['X'], self.tree['right'].node_list['y'])

        if self.tree['left'] is None and self.tree['right'] is None:
            self.node_attrs['is_leaf'] = True
            count_y = {0:0.0,1:0.0}
            for i in y:
                count_y[i] += 1

            if count_y[0] >= count_y[1]:
                self.class_pred = 0
            else:
                self.class_pred = 1

        return


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        
        if(self.node_attrs["is_leaf"] == True):
            return self.class_pred

        class_val = -1

        # If categorical then go left
        if self.node_attrs['is_categorical'] == True: 
            if record[self.node_attrs['split_attr']] == self.node_attrs['split_val']:
                class_val = self.tree['left'].classify(record)
            else:
                class_val = self.tree['right'].classify(record)
        else:
            if record[self.node_attrs['split_attr'] <= self.node_attrs['split_val']]:
                class_val = self.tree['left'].classify(record)
            else:
                class_val = self.tree['right'].classify(record)

        return class_val

    def printPreOrder(self, level=0):

        print("Level: ", level)
        print("Class Pred: ", self.class_pred,", Split Val: ",self.node_attrs['split_val'],", Info gain: ",self.node_attrs['info_gain'])
#        print(self.node_list['X'])

        if self.tree['left'] is not None:
            self.tree['left'].printPreOrder(level+1)

        if self.tree['right'] is not None:
            self.tree['right'].printPreOrder(level+1)

#X = [[3, 'aa', 10],
#    [3, 'aa', 10],
#    [3, 'aa', 10],
#    [3, 'aa', 10],
#    [3, 'aa', 10]]
#    [1, 'bb', 22],
#    [2, 'cc', 28],
#    [5, 'bb', 32],
#    [4, 'cc', 32]]
		 
#y = [1,
#    0,
#    0,
#    0,
#    1]

#root = DecisionTree()
#root.node_list['X'] = X
#root.node_list['y'] = y

#root.learn(X, y)
#print("----Pre-Order----")
#root.printPreOrder()

#print(root.classify([5,'cc',26]))

"""
X = list()
y = list()
XX = list()  # Contains data features and data labels
numerical_cols = set([0,10,11,12,13,15,16,17,18,19,20]) # indices of numeric attributes (columns)

# Loading data set
print("reading hw4-data")
with open("hw4-data.csv") as f:
    next(f, None)

    for line in csv.reader(f, delimiter=","):
        xline = []
        for i in range(len(line)):
            if i in numerical_cols:
                xline.append(ast.literal_eval(line[i]))
            else:
                xline.append(line[i])

        X.append(xline[:-1])
        y.append(xline[-1])
        XX.append(xline[:])

root = DecisionTree()
root.node_list['X'] = X
root.node_list['y'] = y

root.learn(X, y)
print("----Pre-Order----")
root.printPreOrder()
"""
