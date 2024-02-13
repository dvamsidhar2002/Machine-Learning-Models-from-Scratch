# -------------------------- IMPORTING NECESSARY LIBRARIES ------------------------------

import numpy as np
from numpy import log, dot, exp, shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X,Y = make_classification(n_features=4)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=42)

# -------------------------------- STANDARDIZATION ---------------------------------------

def standardize(X_train):
    for i in range(shape(X_train)[1]):
        X_train[:,i] = (X_train[:,i] - np.mean(X_train[:,i]))/np.std(X_train[:,i])

# -------------------------------- F1 Score Calculation ----------------------------------

def F1_score(Y,Y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(Y)):
        if Y[i] == 1 and Y_hat[i] == 1:
            tp+=1
        elif Y[i] == 1 and Y_hat[i] == 0:
            fn+=1
        elif Y[i] == 0 and Y_hat[i] == 1:
            fp+=1
        elif Y[i] == 0 and Y_hat[i] == 0:
            tn+=1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

# ------------------------------- LOGISTIC REGRESSION --------------------------------------

class LogisticRegression:
    def sigmoid(self, z):
        sig = 1/(1+exp(-z))
        return sig

    def initialize(self, X):
        weights = np.zeros((shape(X)[1]+1, 1))
        X = np.c_[np.ones((shape(X)[0], 1)), X]
        return weights, X

    def fit(self,X,y, alpha=0.001, iter=400):
        weights, X = self.initialize(X)
        def cost(theta):
            z = dot(X, theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1 - self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost

        cost_list = np.zeros(iter, )
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list

    def predict(self, X):
        z = dot(self.initialize(X)[1], self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis


standardize(X_train)
standardize(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

pred_y = model.predict(X_test)
train_y = model.predict(X_train)

# Let's see the f1 score for training and testing data

f1_score_train = F1_score(Y_train, train_y)
f1_score_test = F1_score(Y_test, pred_y)

print("F1 Score Training : ",f1_score_train)
print("F1 Score Test : ",f1_score_test)